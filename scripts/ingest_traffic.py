"""批量导入交通运维语料到 traffic_ops_kb 索引。

脚本化直调 kotaemon/ktem 的 Index 与 ingest 管线，不启动任何 Web UI。
复用 FileIndex 的 IndexDocumentPipeline，把 data/traffic_docs/ 下的 Markdown
语料批量导入到新建的 traffic_ops_kb 索引。

用法（PowerShell，工作目录为项目根）：
    .\.venv\Scripts\python.exe scripts\ingest_traffic.py

输出：
    - 每份文件的耗时 / chunk 数 / 失败名单（打印到终端）
    - 汇总结果写入 docs/01_domain_report.md（含导入结果，问答样例由 query_cli 追加）
"""

import os
import sys
import time
from pathlib import Path

# 在导入 ktem 前设置离线环境变量，避免 langchain_mistralai 在 import 时
# 反复重试下载 huggingface tokenizer（国内网络下会卡数十秒）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 确保 libs 在 path 中（脚本从项目根运行时，libs/kotaemon 与 libs/ktem 可导入）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "libs" / "kotaemon"))
sys.path.insert(0, str(ROOT / "libs" / "ktem"))
sys.path.insert(0, str(ROOT))

INDEX_NAME = "traffic_ops_kb"
INDEX_TYPE = "ktem.index.file.FileIndex"
DOCS_DIR = ROOT / "data" / "traffic_docs"
SUPPORTED_EXTS = {".md", ".txt", ".pdf"}

# 无 UI 依赖的占位 UI 类，绕开 gradio 导入（见 headless_ui.py）。
# gradio 4.x 依赖 huggingface_hub 中已被移除的 HfFolder，而本项目
# sentence-transformers/transformers 需要 huggingface_hub>=1.x，二者冲突。
# 入库链路 `get_indexing_pipeline` 完全不需要 UI，故注入占位类跳过 UI 导入。
HEADLESS_UI = "ktem.index.file.headless_ui.FileIndexPage"
HEADLESS_SELECTOR = "ktem.index.file.headless_ui.FileSelector"


def get_or_create_index(mgr):
    """获取 traffic_ops_kb 索引；不存在则创建。

    直接操作 IndexManager（不依赖 BaseApp / gradio）。索引 config 中注入
    FILE_INDEX_UI / FILE_INDEX_SELECTOR_UI 指向 headless 占位类，使
    FileIndex.on_start() 不 import gradio。
    """
    # 已在内存中（启动时从 DB 加载）
    for idx in mgr.indices:
        if idx.name == INDEX_NAME:
            return idx

    # 已在 DB 但未在内存中（本进程首次）→ start_index
    from ktem.index.models import Index
    from sqlmodel import Session, select
    from ktem.db.models import engine

    with Session(engine) as sess:
        row = sess.exec(select(Index).where(Index.name == INDEX_NAME)).one_or_none()

    if row is not None:
        config = dict(row.config or {})
        config.setdefault("FILE_INDEX_UI", HEADLESS_UI)
        config.setdefault("FILE_INDEX_SELECTOR_UI", HEADLESS_SELECTOR)
        return mgr.start_index(row.id, row.name, config, row.index_type)

    # 完全不存在 → 新建
    return mgr.build_index(
        name=INDEX_NAME,
        config={
            "supported_file_types": ".pdf, .txt, .md",
            "private": True,
            "FILE_INDEX_UI": HEADLESS_UI,
            "FILE_INDEX_SELECTOR_UI": HEADLESS_SELECTOR,
        },
        index_type=INDEX_TYPE,
    )


def list_docs(docs_dir: Path) -> list[Path]:
    """列出待入库文档，排除 MANIFEST.md。"""
    files = []
    for p in sorted(docs_dir.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTS and p.name != "MANIFEST.md":
            files.append(p)
    return files


def ingest_file(pipeline, file_path: Path, reindex: bool = True):
    """入库单个文件，返回 (耗时秒, chunk数, 错误信息或None)。"""
    s = time.time()
    n_chunks = 0
    error = None
    gen = pipeline.stream([str(file_path)], reindex=reindex)
    try:
        while True:
            out = next(gen)
            if getattr(out, "channel", "") == "index":
                # index channel 的消息 content 是 dict，含 status
                pass
    except StopIteration as e:
        file_ids, errors, all_docs = e.value
        n_chunks = len(all_docs)
        error = errors[0] if errors and errors[0] else None
    except Exception as e:  # noqa: BLE001
        error = str(e)
    return time.time() - s, n_chunks, error


def main():
    print("=" * 64)
    print("交通运维语料批量入库：", INDEX_NAME)
    print("=" * 64)

    # 不 import ktem.app（其顶部 import gradio 会撞上 HfFolder 错误）。
    # IndexManager 与入库管线只需一个透传的 app 占位对象，用轻量 stub 替代。
    from ktem.index import IndexManager

    class _StubApp:
        pass

    print("[1/3] 实例化索引管理器（无 UI）...")
    mgr = IndexManager(_StubApp())

    print("[2/3] 获取/创建索引", INDEX_NAME, "...")
    idx = get_or_create_index(mgr)
    print(f"      索引 id={idx.id}, 名称={idx.name}")

    settings = {f"index.options.{idx.id}.reader_mode": "default"}
    pipeline = idx.get_indexing_pipeline(settings, user_id="default")

    files = list_docs(DOCS_DIR)
    print(f"[3/3] 待入库文档 {len(files)} 份，开始入库 ...\n")

    results = []
    n_ok, n_fail = 0, 0
    for i, f in enumerate(files, 1):
        dt, n_chunks, err = ingest_file(pipeline, f, reindex=True)
        status = "OK" if err is None else "FAIL"
        if err is None:
            n_ok += 1
        else:
            n_fail += 1
        results.append(
            {"file": f.name, "seconds": round(dt, 3), "chunks": n_chunks, "error": err}
        )
        print(
            f"  [{i:2d}/{len(files)}] {status:4s} {f.name:50s} "
            f"{dt:6.2f}s chunks={n_chunks}"
        )
        if err:
            print(f"           错误: {err}")

    print("\n" + "=" * 64)
    print(f"入库完成：成功 {n_ok} 份，失败 {n_fail} 份")
    print("=" * 64)

    # 输出结果，供 query_cli 或报告使用
    return results


if __name__ == "__main__":
    main()
