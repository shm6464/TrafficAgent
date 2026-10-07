# api/mcp_server.py
"""MCP (Model Context Protocol) 工具服务。

把项目的 RAG 检索、问答、Agent 能力封装为 MCP 工具，供任意支持 MCP 的
客户端（Claude Code / Cursor / 其他 Agent）通过标准协议调用。

基于 mcp.server.fastmcp.FastMCP 实现，暴露三个工具：
    - kb_search：知识库纯检索（返回原文片段 + 出处，不生成）
    - kb_qa：    知识库问答（混合检索 + bge 重排 + LLM 生成）
    - kb_agent： ReAct Agent 问答（多步检索 + 多轮记忆）

为什么用 MCP：
    - MCP 是 2026 年 Agent 岗位 JD 快速上升的硬要求（工具动态注册/调用）；
    - 把「领域知识库」以工具形式开放，使外部 Agent 能"即插即用"地接入
      交通运维知识库，体现「工具化 / 可组合」的架构思维。

复用而非重写：
    - kb_search 复用 api/langgraph_agent.py 的 retrieve（纯检索 + 类别过滤）；
    - kb_qa 复用 api/service.py 的 TrafficQAService.query；
    - kb_agent 复用 api/langgraph_agent.py 的 LangGraphAgent。

硬约束：本文件为「新增模块」，不修改 libs/ 与既有 api/ 源码。

运行（stdio 模式，供 MCP 客户端直接拉起）：
    .\.venv\Scripts\python.exe api\mcp_server.py
"""
import os
import sys
from pathlib import Path

# 确保项目根在 path（供 MCP 客户端从任意 cwd 拉起时能 import api.*）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# ---- 创建 MCP 服务器实例 ----
mcp = FastMCP(
    "traffic-ops-kb",
    instructions=(
        "城市轨道交通运维知识库 MCP 服务。"
        "提供知识库检索、RAG 问答、ReAct Agent 问答三类工具，"
        "服务一线运维与调度人员的规章/技术文档查询。"
    ),
)


# ---------------------------------------------------------------------------
# 工具 1：知识库纯检索
# ---------------------------------------------------------------------------
@mcp.tool()
def kb_search(query: str, category: str = "") -> str:
    """在交通运维知识库中检索相关原文片段（不生成答案）。

    Args:
        query: 检索语句（自然语言，如「信号系统维修分级与周期」）。
        category: 可选类别过滤，取值：search_regulations（规章规范）、
            search_equipment（系统设备）、search_maintenance（检修维护）、
            search_emergency（应急处突）。留空则不分类别过滤。

    Returns:
        命中的原文片段 + 文档出处；未命中返回明确提示。
    """
    from .langgraph_agent import retrieve

    cat = category or None
    return retrieve(query, category=cat, top_k=5)


# ---------------------------------------------------------------------------
# 工具 2：知识库问答（检索 + 生成）
# ---------------------------------------------------------------------------
@mcp.tool()
def kb_qa(question: str, top_k: int = 5) -> str:
    """基于知识库做一次完整 RAG 问答（混合检索 + bge 重排 + LLM 生成）。

    Args:
        question: 用户问题。
        top_k: 召回文档数（1~20，默认 5）。

    Returns:
        带条款号定位与原文出处的答案文本。
    """
    from .service import get_service

    svc = get_service(use_rerank=True, top_k=min(max(top_k, 1), 20))
    result = svc.query(question, top_k=min(max(top_k, 1), 20))
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    if citations:
        cites = "\n".join(f"[{i+1}] {c.get('doc', '-')}" for i, c in enumerate(citations))
        answer = f"{answer}\n\n参考来源：\n{cites}"
    return answer


# ---------------------------------------------------------------------------
# 工具 3：ReAct Agent 问答（多步检索 + 多轮记忆）
# ---------------------------------------------------------------------------
@mcp.tool()
def kb_agent(question: str, session_id: str = "mcp") -> str:
    """用 ReAct Agent 做多步检索问答（支持多轮记忆）。

    Args:
        question: 用户问题。
        session_id: 会话 id（用于多轮上下文，默认 "mcp"）。

    Returns:
        Agent 综合多步检索后给出的最终答案。
    """
    from .langgraph_agent import LangGraphAgent

    agent = LangGraphAgent()
    res = agent.chat(question, session_id=session_id)
    return res.get("answer", "")


# ---------------------------------------------------------------------------
# 入口：stdio 模式（MCP 客户端通过标准输入输出通信）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # stdio 是 MCP 默认传输；如本地调试可设 MCP_TRANSPORT=streamable-http
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
