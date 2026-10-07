"""交通运维知识库问答 CLI（脚本化直调，不起任何服务）。

复用 kotaemon/ktem 的检索与问答链路，对 traffic_ops_kb 索引执行问答，
答案注入交通领域 prompt（要求条款号 + 原文引用 + 文档出处）。

用法（PowerShell，工作目录为项目根）：
    .\.venv\Scripts\python.exe scripts\query_cli.py

默认跑 3 个真实运维问题，打印答案与出处引用，并写入 docs/01_domain_report.md。
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "libs" / "kotaemon"))
sys.path.insert(0, str(ROOT / "libs" / "ktem"))
sys.path.insert(0, str(ROOT))

INDEX_ID = 4  # traffic_ops_kb 对应的 FileIndex id
COLLECTION = f"index_{INDEX_ID}"
TOP_K = 5

# 3 个真实运维问题（覆盖条款级引用、检修周期数值、CBTC 原理）
QUESTIONS = [
    "列车在区间发生故障时的乘客疏散流程是什么？有哪些限速要求？",
    "地铁列车架修和大修的周期分别是多少？",
    "CBTC 降级为点式 ATP 的条件与操作要点？",
]


def build_llm_prompt(question: str, evidence: str) -> str:
    """组装带领域要求的 prompt（与 ktem/prompts/traffic.py 的 QA 模板同构）。"""
    from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT

    return TRAFFIC_QA_TEXT_PROMPT.format(
        lang="中文", context=evidence, question=question
    )


def retrieve(query: str, embedding, vector_store, doc_store, top_k: int = TOP_K):
    """纯向量检索：底层存储 API 直调（绕开框架层 __call__）。"""
    from kotaemon.base import RetrievedDocument

    emb_vec = embedding.run(query)[0].embedding
    _, scores, ids = vector_store.query(embedding=emb_vec, top_k=top_k)
    docs = doc_store.get(ids)
    return [
        RetrievedDocument(**doc.to_dict(), score=score)
        for doc, score in zip(docs, scores)
    ]


def answer(question: str, docs) -> tuple[str, list[str]]:
    """用 llm.invoke 生成答案（绕开框架层），返回 (答案文本, 出处列表)。"""
    from ktem.llms.manager import llms
    from ktem.prompts.traffic import TRAFFIC_SYSTEM_PROMPT
    from kotaemon.base import HumanMessage, SystemMessage

    # 组织 evidence：拼接检索到的文本片段，标注出处
    parts = []
    sources = []
    for i, d in enumerate(docs, 1):
        fn = d.metadata.get("file_name", "-")
        src = fn
        if src not in sources:
            sources.append(src)
        parts.append(f"[出处{i}: {fn}]\n{d.text}")

    evidence = "\n\n".join(parts)
    prompt = build_llm_prompt(question, evidence)

    llm = llms.get_default()
    resp = llm.invoke(
        [SystemMessage(content=TRAFFIC_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return resp.text, sources


def main():
    from ktem.components import get_docstore, get_vectorstore
    from ktem.embeddings.manager import embedding_models_manager

    print("=" * 64)
    print("交通运维知识库问答 CLI（索引: traffic_ops_kb / index_4）")
    print("=" * 64)

    embedding = embedding_models_manager.get_default()
    vector_store = get_vectorstore(COLLECTION)
    doc_store = get_docstore(COLLECTION)
    print("检索链路就绪\n")

    results = []
    for qi, question in enumerate(QUESTIONS, 1):
        print(f">>> 问题 {qi}: {question}\n")
        s = time.time()
        docs = retrieve(question, embedding, vector_store, doc_store)
        t_retrieval = time.time() - s

        print(f"    检索命中 {len(docs)} 个片段（{t_retrieval:.3f}s）:")
        for d in docs:
            print(f"      - {d.metadata.get('file_name','-')}  score={d.score:.4f}")

        s = time.time()
        answer_text, sources = answer(question, docs)
        t_gen = time.time() - s

        print(f"\n    答案（生成 {t_gen:.3f}s）:\n")
        print("    " + answer_text.strip().replace("\n", "\n    "))
        print(f"\n    出处: {'; '.join(sources)}\n")
        print("-" * 64 + "\n")

        results.append(
            {
                "question": question,
                "n_docs": len(docs),
                "retrieval_s": round(t_retrieval, 3),
                "gen_s": round(t_gen, 3),
                "sources": sources,
                "answer": answer_text.strip(),
            }
        )

    return results


if __name__ == "__main__":
    main()
