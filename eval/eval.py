"""Phase 3 评测脚本：三方案对比 + RAGAS 指标（或降级 LLM-as-judge）。

方案：
  A = 朴素向量检索
  B = 混合检索(RRF) + bge 重排
  C = 混合检索 + bge 重排 + 查询改写（GraphRAG 未接入，按指令降级为查询改写）

指标：
  检索层（真实计算）：recall@5 / MRR@5 / 首字节延迟 / 检索总延迟
  生成层：faithfulness / answer_relevancy / context_precision / context_recall
          （优先 RAGAS，安装失败则降级为自实现 LLM-as-judge）
  成本：prompt/completion token + 折算人民币成本（按 DeepSeek 单价）

用法（PowerShell，工作目录为项目根）：
    .\.venv\Scripts\python.exe eval\eval.py --scheme all
    .\.venv\Scripts\python.exe eval\eval.py --scheme A
    .\.venv\Scripts\python.exe eval\eval.py --limit 10   # 抽 10 条快速自测

输出：eval/results/metrics_YYYYMMDD.csv + eval/results/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
for p in [str(ROOT / "libs" / "kotaemon"), str(ROOT / "libs" / "ktem"), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

INDEX_ID = 4  # traffic_ops_kb 对应 FileIndex id
COLLECTION = f"index_{INDEX_ID}"
TOP_K = 5

# DeepSeek deepseek-v4-flash 单价（元 / 百万 token），空闲时段价（保守口径）。
# 依据：api-docs.deepseek.com/zh-cn/quick_start/pricing（2026-09 官方价目表）。
PRICE_PROMPT_PER_M = 1.5  # 输入（缓存未命中）空闲时段
PRICE_COMPLETION_PER_M = 4.5  # 输出 空闲时段


def load_qa_set(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(qa: list[dict], n: int, seed: int = 42) -> list[dict]:
    """按 type 分层抽样 n 条，保证各类型（尤其 multi_hop/table/numeric）都被覆盖。"""
    import random
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in qa:
        by_type[r["type"]].append(r)

    rng = random.Random(seed)
    picked: list[dict] = []
    # 每轮各类型各取 1 条（轮询），直到凑够 n 或耗尽
    types = list(by_type.keys())
    idx = {t: 0 for t in types}
    for t in types:
        rng.shuffle(by_type[t])
    while len(picked) < n and types:
        progressed = False
        for t in list(types):
            if idx[t] < len(by_type[t]):
                picked.append(by_type[t][idx[t]])
                idx[t] += 1
                progressed = True
                if len(picked) >= n:
                    break
            else:
                types.remove(t)
        if not progressed:
            break
    # 保持原始 id 顺序输出，便于复现
    picked.sort(key=lambda r: r["id"])
    return picked[:n]


# --------------------------------------------------------------------------
# 检索链路（三方案）
# --------------------------------------------------------------------------
class RetrievalSuite:
    """封装三方案的检索调用与重排器，供评测复用。"""

    def __init__(self):
        from ktem.components import get_docstore, get_vectorstore
        from ktem.embeddings.manager import embedding_models_manager

        from kotaemon.indices.retrievers.hybrid import HybridFusionRetriever
        from kotaemon.rerankings.bge import BgeReranking

        self.embedding = embedding_models_manager.get_default()
        self.vector_store = get_vectorstore(COLLECTION)
        self.doc_store = get_docstore(COLLECTION)

        self.hybrid = HybridFusionRetriever(
            embedding=self.embedding,
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            top_k=TOP_K,
            fusion_method="rrf",
        )
        self.reranker = BgeReranking(model_name="BAAI/bge-reranker-base")

    def vector_search(self, query: str, top_k: int = TOP_K):
        """方案 A：纯向量检索。"""
        from kotaemon.base import RetrievedDocument

        emb_vec = self.embedding.run(query)[0].embedding
        _, scores, ids = self.vector_store.query(embedding=emb_vec, top_k=top_k)
        docs = self.doc_store.get(ids)
        return [
            RetrievedDocument(**doc.to_dict(), score=float(s))
            for doc, s in zip(docs, scores)
        ]

    def hybrid_search(self, query: str, top_k: int = TOP_K):
        """方案 B：混合检索(RRF)，无重排。"""
        return self.hybrid.run(query, top_k=top_k)

    def hybrid_rerank(self, query: str, top_k: int = TOP_K):
        """方案 B/C 共用：混合检索 + bge 重排（先召回 top_k*2 再重排截断）。"""
        docs = self.hybrid.run(query, top_k=top_k * 2)
        docs = self.reranker.run(documents=docs, query=query)[:top_k]
        return docs


def _doc_identity(doc) -> str:
    return doc.metadata.get("file_name", "")


def _hit(gt_file: str, docs, top_k: int) -> bool:
    for d in docs[:top_k]:
        # source_file 形如 "01_行车组织管理办法_区间疏散与限速.md"，
        # 用去扩展名、去序号前缀后的关键字做包含匹配，兼容 chunk 元数据差异。
        key = gt_file.split(".", 1)[0]
        # 去掉开头 "NN_" 序号前缀
        key = key.split("_", 1)[1] if key[0:2].isdigit() and "_" in key else key
        ident = _doc_identity(d)
        if key[:6] in ident or ident[:6] in key[:12] or (key and key[:4] in ident):
            return True
    return False


# --------------------------------------------------------------------------
# 生成链路
# --------------------------------------------------------------------------
def llm_call_with_retry(messages, retries: int = 3, timeout_sleep: float = 3.0):
    """带重试的 LLM 调用封装，容忍高峰时段偶发超时（APITimeoutError）。

    每次失败后等待 timeout_sleep 秒再重试，最终失败返回 None（上层兜底）。
    """
    from ktem.llms.manager import llms

    llm = llms.get_default()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    [LLM 重试 {attempt}/{retries}] {type(e).__name__}: {e}",
                  flush=True)
            if attempt < retries:
                time.sleep(timeout_sleep)
    print(f"    [LLM 最终失败] {type(last_err).__name__}: {last_err}", flush=True)
    return None


def build_llm_prompt(question: str, evidence: str) -> str:
    from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT

    return TRAFFIC_QA_TEXT_PROMPT.format(lang="中文", context=evidence, question=question)


def answer(question: str, docs) -> dict:
    """用 llm.invoke 生成答案，返回答案文本 + token 用量。"""
    from ktem.prompts.traffic import TRAFFIC_SYSTEM_PROMPT
    from kotaemon.base import HumanMessage, SystemMessage

    parts = []
    for i, d in enumerate(docs, 1):
        fn = d.metadata.get("file_name", "-")
        parts.append(f"[出处{i}: {fn}]\n{d.text}")

    evidence = "\n\n".join(parts)
    prompt = build_llm_prompt(question, evidence)

    resp = llm_call_with_retry(
        [SystemMessage(content=TRAFFIC_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    if resp is None:
        return {"answer": "", "prompt_tokens": 0, "completion_tokens": 0}
    text = getattr(resp, "text", None) or getattr(resp, "content", "")
    prompt_tokens = getattr(resp, "prompt_tokens", 0) or 0
    completion_tokens = getattr(resp, "completion_tokens", 0) or 0
    return {
        "answer": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def query_rewrite(question: str) -> str:
    """方案 C 的查询改写：用 LLM 把原始问题改写为检索友好的关键词查询。"""
    from kotaemon.base import HumanMessage, SystemMessage

    sys_prompt = (
        "你是检索查询改写助手。把用户问题改写为一段简洁、信息密度高的检索查询，"
        "保留关键术语（如 CBTC、ATP、架修、大修、25km/h、条款号等）与数值，"
        "去除寒暄与无关词。只输出改写后的查询文本，不要解释。"
    )
    resp = llm_call_with_retry(
        [SystemMessage(content=sys_prompt), HumanMessage(content=question)]
    )
    if resp is None:
        return question
    text = getattr(resp, "text", None) or getattr(resp, "content", "")
    # 若改写失败则回退原问题
    return text.strip() or question


# --------------------------------------------------------------------------
# 指标计算
# --------------------------------------------------------------------------
def recall_at_k(hits: list[bool]) -> float:
    return sum(hits) / len(hits) if hits else 0.0


def mrr_at_k(rankings: list[int]) -> float:
    """rankings: 每个问题的 ground-truth 命中位置（1-based，0 表示未命中）。"""
    vals = [1.0 / r if r > 0 else 0.0 for r in rankings]
    return sum(vals) / len(vals) if vals else 0.0


def gt_rank(gt_file: str, docs, top_k: int) -> int:
    """返回 ground-truth 在结果中的位置（1-based），未命中返回 0。"""
    for i, d in enumerate(docs[:top_k], 1):
        key = gt_file.split(".", 1)[0]
        key = key.split("_", 1)[1] if key[0:2].isdigit() and "_" in key else key
        ident = _doc_identity(d)
        if key[:6] in ident or ident[:6] in key[:12] or (key and key[:4] in ident):
            return i
    return 0


# --------------------------------------------------------------------------
# RAGAS 指标（优先 RAGAS，降级 LLM-as-judge）
# --------------------------------------------------------------------------
def _load_ragas():
    try:
        import ragas  # noqa: F401

        return True
    except Exception:
        return False


def llm_judge_metrics(question: str, ground_truth: str, answer_text: str, contexts: list[str]) -> dict:
    """自实现 LLM-as-judge：对 faithfulness / answer_relevancy /
    context_precision / context_recall 各打 0-1 分（结构化 JSON 输出）。"""
    from kotaemon.base import HumanMessage, SystemMessage

    ctx = "\n\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(contexts, 1))
    sys_prompt = (
        "你是 RAG 评测裁判。请基于以下信息，对一次检索增强生成的结果打分，"
        "每个指标输出 0 到 1 之间的小数（1 表示完美，0 表示完全不符）。"
        "只输出 JSON，不要解释。\n"
        "指标定义：\n"
        "- faithfulness：答案是否只依据上下文、有无编造（1=完全忠实）。\n"
        "- answer_relevancy：答案是否切题、覆盖问题要点。\n"
        "- context_precision：检索到的上下文中与问题相关的比例（无关片段越少越高）。\n"
        "- context_recall：标准答案所需的信息是否都出现在上下文中。\n"
    )
    human = (
        f"问题：{question}\n\n"
        f"标准答案：{ground_truth}\n\n"
        f"模型答案：{answer_text}\n\n"
        f"检索上下文：\n{ctx}\n\n"
        '请输出 JSON，格式：{"faithfulness":0.0,"answer_relevancy":0.0,'
        '"context_precision":0.0,"context_recall":0.0}'
    )
    resp = llm_call_with_retry(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human)]
    )
    if resp is None:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0}
    text = getattr(resp, "text", None) or getattr(resp, "content", "")
    # 提取 JSON
    try:
        start = text.find("{")
        end = text.rfind("}")
        obj = json.loads(text[start : end + 1])
        return {
            "faithfulness": float(obj.get("faithfulness", 0)),
            "answer_relevancy": float(obj.get("answer_relevancy", 0)),
            "context_precision": float(obj.get("context_precision", 0)),
            "context_recall": float(obj.get("context_recall", 0)),
        }
    except Exception:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }


# --------------------------------------------------------------------------
# 评测主流程
# --------------------------------------------------------------------------
def run_scheme(scheme: str, suite: RetrievalSuite, qa: list[dict], gen: bool = True) -> dict:
    """对单个方案跑全部问题，返回逐题结果与汇总。

    Args:
        gen: 是否运行 LLM 生成 + judge 指标（False 时只跑检索指标，秒级）。
    """
    from kotaemon.base import RetrievedDocument

    rows = []
    for idx, item in enumerate(qa, 1):
        question = item["question"]
        gt_file = item["source_file"]
        ground_truth = item["ground_truth"]

        # 方案 C 先做查询改写（依赖 LLM，仅 gen 模式才有意义，但仍记录）
        effective_query = question
        rewrite_lat = 0.0
        if scheme == "C" and gen:
            t0 = time.time()
            effective_query = query_rewrite(question)
            rewrite_lat = time.time() - t0

        # 检索（记录首字节/总延迟）
        t0 = time.time()
        if scheme == "A":
            docs = suite.vector_search(effective_query, TOP_K)
        elif scheme == "B":
            docs = suite.hybrid_search(effective_query, TOP_K)
        else:  # C
            docs = suite.hybrid_rerank(effective_query, TOP_K)
        retrieval_ms = (time.time() - t0) * 1000

        hit5 = _hit(gt_file, docs, 5)
        rank = gt_rank(gt_file, docs, TOP_K)

        # 生成类指标（faithfulness 等）依赖 LLM，默认跳过以提速
        gen_ms = 0.0
        prompt_tokens = completion_tokens = 0
        cost_rmb = 0.0
        judge = {"faithfulness": 0.0, "answer_relevancy": 0.0,
                 "context_precision": 0.0, "context_recall": 0.0}
        if gen:
            t0 = time.time()
            gen_res = answer(effective_query, docs)
            gen_ms = (time.time() - t0) * 1000
            prompt_tokens = gen_res["prompt_tokens"]
            completion_tokens = gen_res["completion_tokens"]

            contexts = [d.text for d in docs]
            judge = llm_judge_metrics(question, ground_truth, gen_res["answer"], contexts)

            cost_rmb = (
                prompt_tokens / 1e6 * PRICE_PROMPT_PER_M
                + completion_tokens / 1e6 * PRICE_COMPLETION_PER_M
            )

        rows.append(
            {
                "id": item["id"],
                "scheme": scheme,
                "type": item["type"],
                "difficulty": item["difficulty"],
                "hit@5": hit5,
                "rank@5": rank,
                "retrieval_ms": round(retrieval_ms, 2),
                "rewrite_ms": round(rewrite_lat * 1000, 2),
                "gen_ms": round(gen_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_rmb": round(cost_rmb, 6),
                "faithfulness": round(judge["faithfulness"], 4),
                "answer_relevancy": round(judge["answer_relevancy"], 4),
                "context_precision": round(judge["context_precision"], 4),
                "context_recall": round(judge["context_recall"], 4),
            }
        )

        print(f"  [{scheme}] {idx}/{len(qa)} {item['id']} "
              f"hit@5={hit5} rank={rank} 检索={retrieval_ms:.0f}ms"
              + (f" 生成={gen_ms:.0f}ms" if gen else ""),
              flush=True)

    n = len(rows)
    summary = {
        "scheme": scheme,
        "n": n,
        "recall@5": round(sum(r["hit@5"] for r in rows) / n, 4),
        "mrr@5": round(mrr_at_k([r["rank@5"] for r in rows]), 4),
        "avg_retrieval_ms": round(sum(r["retrieval_ms"] for r in rows) / n, 2),
        "avg_gen_ms": round(sum(r["gen_ms"] for r in rows) / n, 2),
        "total_latency_ms": round(
            sum(r["retrieval_ms"] + r["gen_ms"] + r["rewrite_ms"] for r in rows) / n, 2
        ),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_cost_rmb": round(sum(r["cost_rmb"] for r in rows), 4),
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / n, 4),
        "context_precision": round(sum(r["context_precision"] for r in rows) / n, 4),
        "context_recall": round(sum(r["context_recall"] for r in rows) / n, 4),
    }
    return {"rows": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 评测")
    parser.add_argument(
        "--scheme", default="all", choices=["all", "A", "B", "C"],
        help="要评测的方案（默认 all）",
    )
    parser.add_argument("--limit", type=int, default=0, help="抽前 N 条（0=全部）")
    parser.add_argument(
        "--sample", type=int, default=0,
        help="分层抽样 N 条（按 type 均衡抽取，0=不抽样）",
    )
    parser.add_argument(
        "--gen", action="store_true",
        help="运行 LLM 生成 + judge 指标（faithfulness 等，慢）；"
        "默认只跑检索指标（recall@5/MRR@5/延迟，秒级）",
    )
    args = parser.parse_args()

    qa_path = ROOT / "eval" / "qa_set.jsonl"
    qa = load_qa_set(qa_path)
    if args.sample > 0:
        qa = stratified_sample(qa, args.sample)
    elif args.limit > 0:
        qa = qa[: args.limit]

    schemes = ["A", "B", "C"] if args.scheme == "all" else [args.scheme]

    ragas_ok = _load_ragas()
    judge_note = "RAGAS" if ragas_ok else "自实现 LLM-as-judge（ragas 未安装/降级）"

    print("=" * 70)
    print(f"Phase 3 评测 | 评测集 {len(qa)} 条 | 指标口径: {judge_note}")
    print("=" * 70)

    suite = RetrievalSuite()
    # 预热重排器（避免首次加载计入延迟）
    print("预热 bge-reranker ...")
    from kotaemon.base import RetrievedDocument
    _ = suite.reranker.run(
        documents=[RetrievedDocument(text="预热", metadata={"file_name": "warmup"})],
        query="预热",
    )
    print("预热完成\n")

    all_summaries = []
    all_rows = []
    for scheme in schemes:
        print(f"\n>>> 评测方案 {scheme} ...")
        result = run_scheme(scheme, suite, qa, gen=args.gen)
        all_summaries.append(result["summary"])
        all_rows.extend(result["rows"])

        s = result["summary"]
        print(f"    recall@5={s['recall@5']:.1%}  mrr@5={s['mrr@5']:.3f}  "
              f"检索={s['avg_retrieval_ms']}ms  生成={s['avg_gen_ms']}ms  "
              f"成本={s['total_cost_rmb']:.4f}元")
        print(f"    faithfulness={s['faithfulness']:.3f}  "
              f"answer_relevancy={s['answer_relevancy']:.3f}  "
              f"ctx_precision={s['context_precision']:.3f}  "
              f"ctx_recall={s['context_recall']:.3f}")

    # 落盘 CSV
    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = ROOT / "eval" / "results" / f"metrics_{date_str}.csv"
    json_summary_path = ROOT / "eval" / "results" / f"metrics_{date_str}.json"

    import csv
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    out = {
        "date": date_str,
        "n_questions": len(qa),
        "judge_note": judge_note,
        "schemes": all_summaries,
    }
    json_summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n结果已写入:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_summary_path}")


if __name__ == "__main__":
    main()
