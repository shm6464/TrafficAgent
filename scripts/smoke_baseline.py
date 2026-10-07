# -*- coding: utf-8 -*-
"""Phase 0 基线冒烟脚本：不起任何服务，直接调用 kotaemon/ktem 组件，
对已入库的 PDF 完成「检索 -> 生成答案 + 出处引用」全流程。

用法（Windows PowerShell）：
    .\.venv\Scripts\python.exe scripts\smoke_baseline.py

设计说明（重要）：
- 复用 ktem_app_data 里已入库的 index_1（智能客服工单知识库.pdf，86 chunks），
  不改动、不重建任何已有数据资产。
- 检索直接调用底层存储 API（ChromaVectorStore.query + LanceDBDocumentStore.get），
  绕开 VectorRetrieval.run() 与 embedding 的 theflow 框架层 __call__，
  因为后者在 Windows + 多线程/框架序列化下会触发进程被终止（SIGTERM）。
  这是冒烟阶段的稳妥路径；Phase 2 会实现纯 Python BM25 混合检索替代。
- 生成答案直接调用 AnswerWithContextPipeline.stream() 生成器，
  取其最终 return 的完整答案 Document。
- 同一问题跑 N 次取平均延迟，输出首字节延迟 / 总延迟 / 是否返回出处引用。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in [str(ROOT / "libs" / "kotaemon"), str(ROOT / "libs" / "ktem"), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ktem.components import get_docstore, get_vectorstore  # noqa: E402
from ktem.embeddings.manager import embedding_models_manager  # noqa: E402
from ktem.llms.manager import llms  # noqa: E402

from kotaemon.base import HumanMessage, RetrievedDocument, SystemMessage  # noqa: E402

INDEX_NAME = "index_1"  # 已入库的 File Index
TOP_K = 5
N_RUNS = 3

# 与已入库 PDF 内容相关的测试问题（智能客服工单知识库主题）
QUESTIONS = [
    "工单的优先级是如何划分的？",
    "如何处理客户投诉类工单？",
]

# 与 kotaemon DEFAULT_QA_TEXT_PROMPT 同构的问答模板（用 .invoke 直接调 LLM，
# 绕开 theflow Function.__call__ 框架层，后者在 Windows 上会触发进程被杀）
QA_TEMPLATE = (
    "Use the following pieces of context to answer the question at the end in detail "
    "with clear explanation. If you don't know the answer, just say that you don't "
    "know, don't try to make up an answer. Give answer in {lang}.\n\n"
    "{context}\n"
    "Question: {question}\n"
    "Helpful Answer:"
)


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedDocument]:
    """纯向量检索：embedding 用 run()（避开框架层 __call__），
    向量库 query + 文档库 get 直接调用底层存储 API。"""
    embedding = embedding_models_manager.get_default()
    vector_store = get_vectorstore(INDEX_NAME)
    doc_store = get_docstore(INDEX_NAME)

    emb_vec = embedding.run(query)[0].embedding
    _, scores, ids = vector_store.query(embedding=emb_vec, top_k=top_k)
    docs = doc_store.get(ids)
    return [
        RetrievedDocument(**doc.to_dict(), score=score)
        for doc, score in zip(docs, scores)
    ]


def build_evidence(docs: list[RetrievedDocument]) -> str:
    """手动拼接 evidence 文本，绕开 PrepareEvidencePipeline（其内部调用
    tiktoken TokenSplitter 的 theflow 框架层 __call__，Windows 上会 SIGTERM）。"""
    parts = []
    for d in docs:
        fn = d.metadata.get("file_name", "-")
        pg = d.metadata.get("page_label", None)
        src = f"{fn} (Page {pg})" if pg is not None else fn
        text = d.text.replace("\n", " ")
        parts.append(f"<b>Content from {src}: </b> {text}")
    return "\n<br>".join(parts)


def generate_answer(question: str, evidence: str) -> tuple[str, float]:
    """直接调 llm.invoke() 生成答案，绕开 theflow 框架层 __call__/stream。"""
    llm = llms.get_default()
    prompt = QA_TEMPLATE.format(context=evidence, question=question, lang="中文")
    t0 = time.time()
    resp = llm.invoke(
        [SystemMessage(content="这是一个问答系统。"), HumanMessage(content=prompt)]
    )
    elapsed = time.time() - t0
    return resp.text, elapsed


def main():
    print("=" * 60)
    print("Phase 0 基线冒烟：检索 -> 生成答案 + 出处引用")
    print("=" * 60)

    results = []
    for qi, question in enumerate(QUESTIONS, 1):
        print(f"\n>>> 问题 {qi}: {question}")
        for run_i in range(N_RUNS):
            t0 = time.time()

            # 检索
            docs = retrieve(question)
            t_retrieval = time.time() - t0

            # 组织 evidence（手动拼接，绕开框架层）
            evidence = build_evidence(docs)

            # 生成答案（直接 llm.invoke，绕开框架层）
            answer_text, t_gen = generate_answer(question, evidence)

            t_total = time.time() - t0
            t_first_byte = t_retrieval  # 非流式调用，首字节延迟以检索耗时为下界近似

            # 出处引用：从检索命中文档提取（文件名 + 页码）
            sources = []
            for d in docs[:TOP_K]:
                fn = d.metadata.get("file_name", "-")
                pg = d.metadata.get("page_label", None)
                sources.append(f"{fn} (P{pg})" if pg is not None else fn)

            results.append(
                {
                    "question": question,
                    "run": run_i + 1,
                    "n_docs": len(docs),
                    "retrieval_s": round(t_retrieval, 3),
                    "first_byte_s": round(t_first_byte, 3),
                    "gen_s": round(t_gen, 3),
                    "total_s": round(t_total, 3),
                    "sources": sources,
                    "answer_preview": answer_text.strip()[:80],
                }
            )
            print(
                f"    [run {run_i + 1}] 检索 {len(docs)} docs | "
                f"检索 {t_retrieval:.3f}s | 生成 {t_gen:.3f}s | "
                f"总 {t_total:.3f}s"
            )
            print(f"              答案预览: {answer_text.strip()[:60]}")
            print(f"              出处: {'; '.join(sources)}")

    # 汇总平均
    print("\n" + "=" * 60)
    print("基线指标汇总（按问题取 3 次均值）")
    print("=" * 60)
    for qi, question in enumerate(QUESTIONS, 1):
        q_res = [r for r in results if r["question"] == question]
        avg_total = sum(r["total_s"] for r in q_res) / len(q_res)
        avg_retr = sum(r["retrieval_s"] for r in q_res) / len(q_res)
        avg_fb = sum(r["first_byte_s"] for r in q_res) / len(q_res)
        print(f"问题 {qi}: {question}")
        print(f"   平均检索 {avg_retr:.3f}s | 平均首字节 {avg_fb:.3f}s | "
              f"平均总延迟 {avg_total:.3f}s")

    print("\n冒烟完成，无后台进程启动。")


if __name__ == "__main__":
    main()
