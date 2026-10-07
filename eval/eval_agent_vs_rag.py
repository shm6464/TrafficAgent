"""Agent vs RAG 对比评测：证明 ReAct 多工具 Agent 在复合（multi_hop）问题上优于单跳 RAG。

对比对象：
  RAG   = 单跳 service.query（一次检索 + 一次生成）
  Agent = TrafficAgent（ReAct 4 工具 + 多步检索 + token 级流式）

指标（复用 eval.py 的 LLM-as-judge，保证口径一致）：
  faithfulness      答案是否只依据检索内容、有无编造
  answer_relevancy  答案是否切题、覆盖问题要点

用法（PowerShell，项目根）：
  .\\.venv\\Scripts\\python.exe eval\\eval_agent_vs_rag.py --limit 10   # 抽 10 条快速
  .\\.venv\\Scripts\\python.exe eval\\eval_agent_vs_rag.py              # 全部 23 条

输出：eval/results/agent_vs_rag_YYYYMMDD.csv + .json + 控制台汇总。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
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

# 复用 eval.py 的评测组件（load_qa_set / llm_judge_metrics / stratified_sample）
_spec = importlib.util.spec_from_file_location("eval_mod", ROOT / "eval" / "eval.py")
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
load_qa_set = _eval_mod.load_qa_set
llm_judge_metrics = _eval_mod.llm_judge_metrics
stratified_sample = _eval_mod.stratified_sample


def run_rag(question: str) -> tuple[str, list[str]]:
    """单跳 RAG：service.query 一次检索 + 一次生成。"""
    try:
        from api.service import get_service

        res = get_service().query(question, top_k=5)
        ctx = [c.get("text", "") for c in res.get("citations", [])]
        return res.get("answer", ""), ctx
    except Exception as e:  # noqa: BLE001
        print(f"    [RAG 异常] {type(e).__name__}: {e}", flush=True)
        return "（生成失败）", []


def _extract_observations(trace: str) -> list[str]:
    """从 Agent 的 ReAct trace 里提取 Observation（真实检索结果）作为 judge 的 context。"""
    obs = re.findall(r"Observation:\s*(.*?)(?=\nThought:|\nAction:|\Z)", trace, re.S)
    return [o.strip() for o in obs if o.strip()]


def run_agent(question: str, agent) -> tuple[str, list[str]]:
    """ReAct Agent：多步检索 + 生成。"""
    try:
        res = agent.run(question)
        ctx = _extract_observations(res.get("trace", ""))
        return res.get("answer", ""), ctx
    except Exception as e:  # noqa: BLE001
        print(f"    [Agent 异常] {type(e).__name__}: {e}", flush=True)
        return "（生成失败）", []


def main():
    parser = argparse.ArgumentParser(description="Agent vs RAG 对比评测")
    parser.add_argument("--limit", type=int, default=10, help="抽前 N 条 multi_hop（0=全部 23 条）")
    parser.add_argument("--sample", type=int, default=0, help="按 difficulty 分层抽样 N 条")
    args = parser.parse_args()

    qa = load_qa_set(ROOT / "eval" / "qa_set.jsonl")
    multi_hop = [r for r in qa if r["type"] == "multi_hop"]
    if args.sample > 0:
        multi_hop = stratified_sample(multi_hop, args.sample)
    elif args.limit > 0:
        multi_hop = multi_hop[: args.limit]

    print("预热（加载 embedding / bge-reranker / LLM）...", flush=True)
    from api.service import get_service

    get_service()
    from api.agent import TrafficAgent

    agent = TrafficAgent(max_iter=5, top_k=5)
    print("预热完成\n", flush=True)

    print(f"Agent vs RAG 评测 | multi_hop {len(multi_hop)} 条\n", flush=True)
    rows = []
    for i, item in enumerate(multi_hop, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(multi_hop)}] {item['id']} {q[:44]}", flush=True)

        t0 = time.time()
        rag_answer, rag_ctx = run_rag(q)
        rag_ms = (time.time() - t0) * 1000
        rag_judge = llm_judge_metrics(q, gt, rag_answer, rag_ctx)
        print(f"    RAG    {rag_ms:6.0f}ms  faith={rag_judge['faithfulness']:.2f}  relev={rag_judge['answer_relevancy']:.2f}", flush=True)

        t0 = time.time()
        agent_answer, agent_ctx = run_agent(q, agent)
        agent_ms = (time.time() - t0) * 1000
        agent_judge = llm_judge_metrics(q, gt, agent_answer, agent_ctx)
        print(f"    Agent  {agent_ms:6.0f}ms  faith={agent_judge['faithfulness']:.2f}  relev={agent_judge['answer_relevancy']:.2f}", flush=True)

        rows.append(
            {
                "id": item["id"],
                "difficulty": item["difficulty"],
                "question": q,
                "rag_faithfulness": round(rag_judge["faithfulness"], 4),
                "rag_answer_relevancy": round(rag_judge["answer_relevancy"], 4),
                "agent_faithfulness": round(agent_judge["faithfulness"], 4),
                "agent_answer_relevancy": round(agent_judge["answer_relevancy"], 4),
                "rag_ms": round(rag_ms, 2),
                "agent_ms": round(agent_ms, 2),
                "rag_answer": rag_answer,
                "agent_answer": agent_answer,
            }
        )

    n = len(rows)

    def avg(k):
        return round(sum(r[k] for r in rows) / n, 4) if n else 0.0

    summary = {
        "n": n,
        "rag_faithfulness": avg("rag_faithfulness"),
        "agent_faithfulness": avg("agent_faithfulness"),
        "rag_answer_relevancy": avg("rag_answer_relevancy"),
        "agent_answer_relevancy": avg("agent_answer_relevancy"),
        "faithfulness_gain": round(avg("agent_faithfulness") - avg("rag_faithfulness"), 4),
        "relevancy_gain": round(avg("agent_answer_relevancy") - avg("rag_answer_relevancy"), 4),
    }

    print("\n" + "=" * 64)
    print(f"faithfulness      RAG {summary['rag_faithfulness']:.3f}  ->  Agent {summary['agent_faithfulness']:.3f}  ({summary['faithfulness_gain']:+.3f})")
    print(f"answer_relevancy  RAG {summary['rag_answer_relevancy']:.3f}  ->  Agent {summary['agent_answer_relevancy']:.3f}  ({summary['relevancy_gain']:+.3f})")
    print("=" * 64)

    date = datetime.now().strftime("%Y%m%d")
    csv_path = ROOT / "eval" / "results" / f"agent_vs_rag_{date}.csv"
    json_path = ROOT / "eval" / "results" / f"agent_vs_rag_{date}.json"

    import csv as csv_mod

    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入:\n  CSV:  {csv_path}\n  JSON: {json_path}")


if __name__ == "__main__":
    main()
