"""Phase 2 检索对比脚本：纯向量 vs 混合(RRF) vs 混合+重排(bge)。

不起任何服务，脚本直调检索链路，对 10 个运维问题统计 recall@5。
输出写入 docs/02_retrieval_report.md 所需的数据。

用法：
    .venv\\Scripts\\python.exe scripts\\smoke_retrieval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 确保 libs 与项目根在 path 中
ROOT = Path(__file__).resolve().parent.parent
for p in [str(ROOT / "libs" / "kotaemon"), str(ROOT / "libs" / "ktem"), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ktem.components import get_docstore, get_vectorstore  # noqa: E402
from ktem.embeddings.manager import embedding_models_manager  # noqa: E402
from kotaemon.base import RetrievedDocument  # noqa: E402
from kotaemon.indices.retrievers.hybrid import (  # noqa: E402
    HybridFusionRetriever,
)
from kotaemon.rerankings.bge import BgeReranking  # noqa: E402

INDEX_NAME = "index_4"  # traffic_ops_kb 的 collection（FileIndex 用 index_{id}）
TOP_K = 5

# 12 个测试问题：(问题, ground-truth 文档文件名关键字)
# ground-truth 由人工从语料原文确认，用于统计 recall@k。
# 前 6 个为"简单"（唯一主题），后 6 个为"困难"（跨专业同名概念/语义混淆），
# 用于拉开三种方案的区分度。
QUESTIONS = [
    # --- 简单问题：主题唯一，向量即可命中 ---
    ("列车在区间发生故障时乘客如何疏散、限速多少？", "01_行车组织管理办法"),
    ("CBTC 信号系统 ATP 层级的职责是什么？", "03_CBTC信号系统原理"),
    ("地铁牵引供电系统采用什么电压等级？", "06_牵引供电系统原理"),
    ("发生火灾时应急处置流程是什么？", "15_火灾应急处置与消防安全"),
    ("列车再生制动如何实现能量回馈？", "12_列车再生制动与能量回馈"),
    ("列车定位测速有哪些技术手段？", "10_列车定位与测速技术"),
    # --- 困难问题：跨专业同名概念，需关键词精确匹配 ---
    ("信号系统的大修周期不超过多少年？", "13_信号系统维修分级与周期"),
    ("通信系统的大修周期不超过多少年？", "14_通信系统维修分级与周期"),
    ("车辆架修 A 与架修 B 的区别是什么？", "04_车辆检修修程与周期"),
    ("设施设备维护规程应报哪个部门备案？", "02_设施设备运行维护管理办法"),
    ("信号系统与通信系统维修分级有何异同？", "13_信号系统维修分级与周期"),
    ("站台门系统与列车门如何联动控制？", "08_站台门屏蔽门系统"),
]


def _doc_identity(doc: RetrievedDocument) -> str:
    """返回文档的 file_name（用于匹配 ground-truth）。"""
    return doc.metadata.get("file_name", "")


def _hit(gt_keyword: str, docs: list[RetrievedDocument], top_k: int = TOP_K) -> bool:
    """判断 ground-truth 文档是否出现在 Top-K 结果中。"""
    for d in docs[:top_k]:
        if gt_keyword in _doc_identity(d):
            return True
    return False


def build_vector_retriever():
    """纯向量检索（方案 A）。"""
    embedding = embedding_models_manager.get_default()
    vector_store = get_vectorstore(INDEX_NAME)
    doc_store = get_docstore(INDEX_NAME)
    return embedding, vector_store, doc_store


def vector_search(embedding, vector_store, doc_store, query, top_k=TOP_K):
    emb_vec = embedding.run(query)[0].embedding
    _, scores, ids = vector_store.query(embedding=emb_vec, top_k=top_k)
    docs = doc_store.get(ids)
    return [
        RetrievedDocument(**doc.to_dict(), score=float(s))
        for doc, s in zip(docs, scores)
    ]


def main():
    print("=" * 70)
    print("Phase 2 检索对比：A 纯向量 | B 混合(RRF) | C 混合+重排(bge)")
    print("=" * 70)

    embedding, vector_store, doc_store = build_vector_retriever()

    # 方案 B / C 共用混合检索器
    hybrid = HybridFusionRetriever(
        embedding=embedding,
        vector_store=vector_store,
        doc_store=doc_store,
        top_k=TOP_K,
        fusion_method="rrf",
    )

    # 方案 C 的重排器
    reranker = BgeReranking(model_name="BAAI/bge-reranker-base")
    # 预热模型（避免首次加载计入延迟）
    print("\n预热 bge-reranker 模型...")
    _ = reranker.run(
        documents=[
            RetrievedDocument(text="预热", metadata={"file_name": "warmup"})
        ],
        query="预热",
    )
    print("预热完成\n")

    results = []
    for qi, (question, gt) in enumerate(QUESTIONS, 1):
        row = {"qid": qi, "question": question, "gt": gt}

        # 方案 A：纯向量
        t0 = time.time()
        a_docs = vector_search(embedding, vector_store, doc_store, question)
        row["A_hit5"] = _hit(gt, a_docs, 5)
        row["A_hit3"] = _hit(gt, a_docs, 3)
        row["A_hit1"] = _hit(gt, a_docs, 1)
        row["A_ms"] = round((time.time() - t0) * 1000, 1)
        row["A_top"] = [_doc_identity(d) for d in a_docs[:3]]

        # 方案 B：混合 RRF
        t0 = time.time()
        b_docs = hybrid.run(question, top_k=TOP_K)
        row["B_hit5"] = _hit(gt, b_docs, 5)
        row["B_hit3"] = _hit(gt, b_docs, 3)
        row["B_hit1"] = _hit(gt, b_docs, 1)
        row["B_ms"] = round((time.time() - t0) * 1000, 1)
        row["B_top"] = [_doc_identity(d) for d in b_docs[:3]]

        # 方案 C：混合 + bge 重排
        t0 = time.time()
        c_docs = hybrid.run(question, top_k=TOP_K * 2)  # 先召回更多
        c_docs = reranker.run(documents=c_docs, query=question)[:TOP_K]
        row["C_hit5"] = _hit(gt, c_docs, 5)
        row["C_hit3"] = _hit(gt, c_docs, 3)
        row["C_hit1"] = _hit(gt, c_docs, 1)
        row["C_ms"] = round((time.time() - t0) * 1000, 1)
        row["C_top"] = [_doc_identity(d) for d in c_docs[:3]]

        results.append(row)

        print(f"[Q{qi}] {question}")
        print(f"      GT: {gt}")
        print(f"      A(向量)       hit5={row['A_hit5']} hit1={row['A_hit1']}  {row['A_ms']}ms")
        print(f"      B(混合)       hit5={row['B_hit5']} hit1={row['B_hit1']}  {row['B_ms']}ms")
        print(f"      C(混合+重排)  hit5={row['C_hit5']} hit1={row['C_hit1']}  {row['C_ms']}ms")
        # 打印三种方案 Top1 以便失败案例对比
        if not (row["A_hit1"] and row["B_hit1"] and row["C_hit1"]):
            print(f"        A Top1: {row['A_top'][0] if row['A_top'] else '-'}")
            print(f"        B Top1: {row['B_top'][0] if row['B_top'] else '-'}")
            print(f"        C Top1: {row['C_top'][0] if row['C_top'] else '-'}")

    # 汇总 recall@k
    n = len(results)

    def recall(key):
        return sum(r[key] for r in results) / n

    def avg_ms(key):
        return sum(r[key] for r in results) / n

    print("\n" + "=" * 70)
    print("汇总（recall@k）")
    print("=" * 70)
    for label, prefix in [("A 纯向量", "A"), ("B 混合(RRF)", "B"), ("C 混合+重排", "C")]:
        print(
            f"{label:14s} recall@1={recall(prefix+'_hit1'):.1%}  "
            f"recall@3={recall(prefix+'_hit3'):.1%}  "
            f"recall@5={recall(prefix+'_hit5'):.1%}  "
            f"平均 {avg_ms(prefix+'_ms'):.1f}ms"
        )

    # 落盘 JSON（供报告引用）
    out = {
        "results": results,
        "summary": {
            "n_questions": n,
            "A_recall@1": recall("A_hit1"),
            "A_recall@3": recall("A_hit3"),
            "A_recall@5": recall("A_hit5"),
            "B_recall@1": recall("B_hit1"),
            "B_recall@3": recall("B_hit3"),
            "B_recall@5": recall("B_hit5"),
            "C_recall@1": recall("C_hit1"),
            "C_recall@3": recall("C_hit3"),
            "C_recall@5": recall("C_hit5"),
            "A_avg_ms": avg_ms("A_ms"),
            "B_avg_ms": avg_ms("B_ms"),
            "C_avg_ms": avg_ms("C_ms"),
        },
    }
    out_path = ROOT / "docs" / "02_retrieval_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
