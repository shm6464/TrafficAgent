# scripts/test_langgraph.py
"""LangGraph 版 ReAct Agent 冒烟测试（进程内，不启动服务）。

验证 api/langgraph_agent.py 的状态机能正常构建、单次问答能跑通。
用法（PowerShell，工作目录为项目根）：
    .\.venv\Scripts\python.exe scripts\test_langgraph.py

输出：图结构、工具清单、一次真实问答的答案与耗时。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.langgraph_agent import (
    LangGraphAgent,
    _build_graph,
    TOOL_NAMES,
    _should_continue,
)


def test_graph_builds():
    g = _build_graph()
    nodes = list(g.get_graph().nodes.keys())
    assert "__start__" in nodes and "agent" in nodes and "tools" in nodes and "__end__" in nodes
    print("[OK] 图构建成功，节点：", nodes)


def test_tool_names():
    assert len(TOOL_NAMES) == 4
    print("[OK] 工具清单：", TOOL_NAMES)


def test_conditional_logic():
    """条件边逻辑：无工具调用应返回 end（纯函数级验证，不调 LLM）。"""
    from langchain_core.messages import AIMessage, HumanMessage

    # 有 tool 调用 → tools
    s1 = {"messages": [AIMessage(content='{"tool": "search_equipment", "input": "CBTC"}')], "iterations": 0}
    assert _should_continue(s1) == "tools"
    # 无 tool（answer）→ end
    s2 = {"messages": [AIMessage(content='{"answer": "最终答案"}')], "iterations": 0}
    assert _should_continue(s2) == "end"
    # 超迭代 → end
    s3 = {"messages": [AIMessage(content='{"tool": "search_equipment", "input": "x"}')], "iterations": 99}
    assert _should_continue(s3) == "end"
    print("[OK] 条件边逻辑正确（tools / end / 超迭代）")


def test_e2e():
    """端到端：真实 LLM + 检索。首次运行会加载 bge-reranker，较慢。"""
    agent = LangGraphAgent()
    t0 = time.time()
    res = agent.run("信号系统维修分级与周期")
    dt = round(time.time() - t0, 2)
    print(f"[OK] 端到端问答（{dt}s）：")
    print("    答案：", res["answer"][:200])
    assert res["answer"], "答案不应为空"


if __name__ == "__main__":
    test_graph_builds()
    test_tool_names()
    test_conditional_logic()
    test_e2e()
    print("\n全部测试通过 ✅")
