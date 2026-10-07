"""延迟与缓存收益基准测试（Phase 4 工程化）。

跑 20 个真实问题，输出：
- 检索延迟 / 生成延迟 / 总延迟的 P50、P95
- 缓存开/关（LLM 缓存 + embedding 缓存）两轮的对比
- token 成本与缓存命中率

用法（PowerShell，项目根）：
    .\\.venv\\Scripts\\python.exe scripts\\bench_latency.py
    .\\.venv\\Scripts\\python.exe scripts\\bench_latency.py --cache-off   # 关闭缓存对照

说明：
- 第一轮「缓存关」是冷启动（每次真实调 LLM + embedding）；
- 第二轮「缓存开」时，重复问题命中 LLM 缓存、embedding 缓存，体现收益。
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in [ROOT / "libs" / "kotaemon", ROOT / "libs" / "ktem", ROOT]:
    sys.path.insert(0, str(p))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 20 个真实交通运维问题（来自 eval/qa_set.jsonl 的典型题）
QUESTIONS = [
    "列车在区间发生故障时，乘客疏散的流程是什么？",
    "CBTC 降级为点式 ATP 的条件与操作要点？",
    "信号系统的大修周期是多少年？",
    "车辆架修的周期和里程上限是多少？",
    "火灾发生时，车站的应急处置流程？",
    "牵引供电系统的电压等级有哪些？",
    "站台门系统的主要功能是什么？",
    "轨道设施的养护维修分为哪几级？",
    "通信系统的维修分级与周期？",
    "再生制动能量是如何回馈的？",
    "运营突发事件分为哪几级？",
    "CBTC 与固定闭塞的本质区别？",
    "列车定位测速用到了哪些传感器？",
    "供电系统实时监控的关键部位有哪些？",
    "区间疏散时列车限速是多少？",
    "自动售检票系统 AFC 的组成？",
    "设施设备维护有哪些分级要求？",
    "应急预案演练有哪些要求？",
    "行车组织管理办法第三十五条规定了什么？",
    "列车检修的列检周期是多少？",
]


def pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = min(len(data) - 1, int(len(data) * p))
    return data[idx]


def run_round(cache_on: bool, n: int):
    from api.service import TrafficQAService

    svc = TrafficQAService(use_rerank=True, top_k=5)

    latencies = []
    total_cost = 0.0
    for i, q in enumerate(QUESTIONS[:n], 1):
        t0 = time.time()
        result = svc.query(q)
        lat = result["latency_ms"]
        latencies.append(lat)
        total_cost += result.get("cost_rmb", 0.0)
        print(
            f"  [{i:2d}/{n}] {lat:8.1f}ms  cost={result.get('cost_rmb', 0):.4f}  "
            f"{q[:20]}..."
        )

    print(f"\n  缓存状态: {'开' if cache_on else '关'}")
    print(f"  P50 = {pct(latencies, 0.50):.1f}ms  P95 = {pct(latencies, 0.95):.1f}ms")
    print(f"  平均 = {statistics.mean(latencies):.1f}ms  总成本 = {total_cost:.4f} 元")
    print(f"  缓存命中率: {svc.cache_stats()}")
    return {
        "p50": pct(latencies, 0.50),
        "p95": pct(latencies, 0.95),
        "mean": statistics.mean(latencies),
        "cost": total_cost,
        "cache_stats": svc.cache_stats(),
    }


def _set_cache(enabled: bool):
    """直接操作 theflow settings 对象属性，控制双层缓存开关。"""
    from theflow.settings import settings

    settings.USE_CACHE = enabled
    settings.USE_LLM_CACHE = enabled
    settings.USE_EMBEDDING_CACHE = enabled


def _clear_llm_cache():
    """清空 LLM 响应缓存目录，保证「冷启动」轮纯净（diskcache 跨进程持久）。"""
    from theflow.settings import settings

    cache_dir = Path(getattr(settings, "KH_APP_DATA_DIR", ".")) / "cache" / "llm_cache"
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        print(f"  已清空 LLM 缓存目录: {cache_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="问题数（默认 20）")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="不清空缓存（默认清空，保证冷/热对比纯净）",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("交通运维知识库 延迟基准测试")
    print("=" * 64)

    # 第一轮：冷启动（清空 LLM 缓存，真实调用）
    print("\n[第 1 轮] 冷启动（清空 LLM 缓存，真实调用）...")
    if not args.keep_cache:
        _clear_llm_cache()
    _set_cache(True)
    cold = run_round(cache_on=False, n=args.n)

    # 第二轮：热缓存（重复问题命中 LLM 缓存）
    print("\n[第 2 轮] 热缓存（重复问题命中 LLM 缓存）...")
    import api.service as svc_mod

    svc_mod._service = None
    warm = run_round(cache_on=True, n=args.n)

    print("\n" + "=" * 64)
    print("对比汇总")
    print("=" * 64)
    print(f"{'指标':<14}{'缓存关(冷)':<16}{'缓存开(热)':<16}{'提升'}")
    print(f"{'P50(ms)':<14}{cold['p50']:<16.1f}{warm['p50']:<16.1f}"
          f"{_improve(cold['p50'], warm['p50'])}")
    print(f"{'P95(ms)':<14}{cold['p95']:<16.1f}{warm['p95']:<16.1f}"
          f"{_improve(cold['p95'], warm['p95'])}")
    print(f"{'均值(ms)':<14}{cold['mean']:<16.1f}{warm['mean']:<16.1f}"
          f"{_improve(cold['mean'], warm['mean'])}")
    print(f"{'成本(元)':<14}{cold['cost']:<16.4f}{warm['cost']:<16.4f}"
          f"{_improve(cold['cost'], warm['cost'])}")
    print("=" * 64)


def _improve(cold: float, warm: float) -> str:
    if cold <= 0:
        return ""
    pct_change = (cold - warm) / cold * 100
    return f"{pct_change:+.1f}%"


if __name__ == "__main__":
    main()
