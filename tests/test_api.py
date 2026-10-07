"""Phase 4 API 测试：进程内用 fastapi.testclient.TestClient 验证接口。

不启动任何真实服务，直接调用 /healthz 与 /query，断言返回 JSON 结构
符合接口定义并打印真实样例。运行（PowerShell，项目根）：

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_api.py -v

注意：/query 会真实调用 LLM（DeepSeek），单次可能 10-30 秒；
若只想测结构可设置环境变量 SKIP_LLM_QUERY=1 跳过生成断言。
"""

import os
import sys
from pathlib import Path

# 确保项目根与 libs 在 path 中
ROOT = Path(__file__).resolve().parent.parent
for p in [ROOT / "libs" / "kotaemon", ROOT / "libs" / "ktem", ROOT]:
    sys.path.insert(0, str(p))

# 离线环境变量（避免 import 时下载 tokenizer）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)

SKIP_LLM = os.environ.get("SKIP_LLM_QUERY", "0") == "1"


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print(f"[healthz] {data}")


def test_query_structure():
    if SKIP_LLM:
        print("[query] SKIP_LLM_QUERY=1，跳过真实 LLM 调用")
        return
    payload = {
        "question": "列车在区间发生故障时，乘客疏散的流程是什么？",
        "top_k": 5,
        "use_rerank": True,
    }
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # 断言结构符合接口定义
    assert isinstance(data["answer"], str)
    assert isinstance(data["citations"], list)
    assert isinstance(data["latency_ms"], (int, float))
    assert isinstance(data["tokens"], dict)
    assert "trace_id" in data

    print(f"[query] answer[:80] = {data['answer'][:80]}")
    print(f"[query] citations = {len(data['citations'])} 条")
    print(f"[query] latency_ms = {data['latency_ms']}")
    print(f"[query] tokens = {data['tokens']}")
    print(f"[query] trace_id = {data['trace_id']}")

    # 至少返回 1 条引用（语料已建库）
    assert len(data["citations"]) >= 1


def test_query_reject_empty_question():
    resp = client.post("/query", json={"question": ""})
    # 空问题会被 pydantic 接受（str），但语义上应返回结果或 422（由 min_length 控制）
    # 这里只断言接口不会 500 崩溃
    assert resp.status_code in (200, 422)


if __name__ == "__main__":
    test_healthz()
    test_query_structure()
    test_query_reject_empty_question()
    print("\n=== 全部测试通过 ===")
