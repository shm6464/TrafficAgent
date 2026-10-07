# api/langgraph_agent.py
"""LangGraph 版 ReAct Agent。

用 LangGraph StateGraph 重写 api/agent.py 里手写的 while 循环 ReAct，
把「思考 → 检索 → 观察 → 再思考 → 作答」建模为显式状态机：

    START → agent（LLM 决策）──条件边──┬─ 需检索 → tools（纯检索，不生成）
           ▲                            │                  │
           └────────────────────────────┴──────────────────┘
                 信息足够 / 超迭代 → END（提取最终答案）

为什么手写 StateGraph 而不是 create_react_agent：
    - 本项目 LLM 是 kotaemon 封装的 ChatOpenAI，无 bind_tools（原生 tool calling
      不可用），create_react_agent 依赖 bind_tools，走不通；
    - 手写状态机更贴合简历叙事：显式状态 + 条件边 + 节点编排，能讲清
      LangGraph 的核心机制。

设计要点（与 api/agent.py 的关键差异）：
    - tools 节点只做「检索 + 返回原文片段」，不调 LLM 生成答案——生成统一
      收敛到 agent 节点的最终作答，避免「检索一次 + 生成一次」的双重 LLM 调用；
    - 复用 api/service.py 的 HybridFusionRetriever（混合检索 + bge 重排），
      按 api/agent.py 的 _classify_file 做文档类别过滤。

复用而非重写：
    - 检索链路：api/service.py 的 get_service().retriever；
    - 文档分类：api/agent.py 的 TOOLS / _classify_file；
    - 多轮记忆：api/memory_store.py（支持 redis 降级内存）；
    - 领域 prompt：ktem.prompts.traffic。

硬约束：本文件为「新增模块」，不修改 api/agent.py、api/service.py 与
libs/ 下任何源码，满足 INSTRUCTIONS.md「不破坏现有能力」。
"""
from __future__ import annotations

import json
import re
import time
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from kotaemon.base import HumanMessage as KHumanMessage
from kotaemon.base import SystemMessage as KSystemMessage
from kotaemon.base import AIMessage as KAIMessage

from ktem.llms.manager import llms
from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT, TRAFFIC_SYSTEM_PROMPT

# 复用 api/agent.py 的工具分类（不重复定义）；检索走 api/service.py 的 retriever
from .agent import TOOLS, _classify_file  # noqa: F401
from .service import get_service
from .memory_store import get_memory_store

# ---- 多轮会话存储（可插拔：默认内存，MEMORY_BACKEND=redis 可切换）----
_store = get_memory_store()

MAX_TOOL_ITERATIONS = 5     # 最多调 5 次工具，防死循环
RETRIEVE_TOP_K = 5          # 每次工具检索召回数


# ---------------------------------------------------------------------------
# 1. 图状态定义
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """LangGraph 共享状态：跨节点传递的消息列表与迭代计数。

    messages 用 add_messages 归约器合并（新消息追加、ToolMessage 按
    tool_call_id 覆盖占位），是 LangGraph 状态机的标准写法。
    """

    messages: Annotated[list, add_messages]
    iterations: int          # 已执行的工具轮数（防死循环）


# ---------------------------------------------------------------------------
# 2. 纯检索工具（不调 LLM）
# ---------------------------------------------------------------------------
def retrieve(query: str, category: str = None, top_k: int = RETRIEVE_TOP_K) -> str:
    """执行一次纯检索，返回原文片段 + 出处，不生成答案。

    复用 service 的 HybridFusionRetriever（向量+BM25 混合 + bge 重排），
    若指定 category 则按文档类别过滤（复用 api/agent.py 的 _classify_file）。
    """
    svc = get_service()
    docs = svc.retriever.run(query, top_k=max(top_k * 4, 20))
    if category:
        docs = [d for d in docs if _classify_file(d.metadata.get("file_name", "")) == category]
    if not docs:
        return "未在知识库中检索到相关内容。"
    evidence = "\n\n".join(
        f"[出处{i}: {d.metadata.get('file_name', '-')}]\n{d.text}"
        for i, d in enumerate(docs[:top_k], 1)
    )
    return evidence


TOOL_DESCRIPTIONS = "\n".join(f"- {name}：{info['desc']}" for name, info in TOOLS.items())
TOOL_NAMES = list(TOOLS.keys())


# ---------------------------------------------------------------------------
# 3. 节点实现
# ---------------------------------------------------------------------------
def _to_kotaemon_message(msg):
    """把 langchain_core 消息转换成 kotaemon.base 消息类型。

    kotaemon 的 ChatOpenAI.invoke 只认自己的 BaseMessage（prepare_message
    内部调用 to_openai_format），langchain 的 SystemMessage 没有该方法，
    直接传入会抛 AttributeError。故在此做显式类型转换。
    """
    content = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(msg, SystemMessage):
        return KSystemMessage(content=content)
    if isinstance(msg, ToolMessage):
        # 工具观察结果以 user 角色喂回模型（无原生 tool role 支持）
        return KHumanMessage(content=f"[检索结果]\n{content}")
    if isinstance(msg, AIMessage):
        return KAIMessage(content=content)
    return KHumanMessage(content=content)


def _parse_json(raw: str):
    """健壮地解析 LLM 输出的 JSON（容忍 markdown 代码块 / 前后缀说明 / 裸换行）。

    glm-4.5-air 等模型对「只输出 JSON」的约束遵守不稳定，常见三类问题：
        1. 用 ```json ... ``` 代码块包裹；
        2. JSON 前后带一句说明文字；
        3. answer 字段内是「真实换行符」而非 \\n 转义（markdown 答案必然含换行）。
    裸 json.loads 会因这些直接抛异常导致整轮报废。故：
        - 先剥离 ``` 围栏；
        - 用 strict=False 解析（允许字符串内的裸控制字符如换行/tab）；
        - 仍失败则花括号截取首个 JSON 对象再解析。全部失败返回 None。
    """
    if not raw:
        return None
    text = raw.strip()
    # 1) 剥离 markdown 代码块围栏 ```json ... ``` / ``` ... ```
    if text.startswith("```"):
        text = text.strip("`")
        # 去掉可能紧跟的语言标记（json / JSON / 空）
        nl = text.find("\n")
        if nl != -1:
            head = text[:nl].strip().lower()
            if head in ("json", ""):
                text = text[nl + 1 :]
        # 再去掉结尾的 ``` 残留
        text = text.strip().strip("`").strip()
    # 2) 直接解析（strict=False 容忍裸控制字符）
    try:
        return json.loads(text, strict=False)
    except Exception:
        pass
    # 3) 用花括号截取首个完整 JSON 对象（兼容前后缀文字）
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1], strict=False)
        except Exception:
            pass
    return None


def _chat(messages: list) -> str:
    """调 LLM，带 3 次重试容错（DeepSeek/百炼高峰时段偶发超时）。

    用 stream 流式调用并拼接完整输出。原因：阿里云百炼的 glm-4.5-air
    等免费额度模型只支持 stream 模式，非流式 invoke（stream=False）会
    被拒绝（BadRequestError: only support stream mode）。故统一走
    llm.stream 拼接，对非流式模型同样兼容。
    """
    k_messages = [_to_kotaemon_message(m) for m in messages]
    for attempt in range(1, 4):
        try:
            parts = []
            for chunk in llms.get_default().stream(k_messages):
                text = getattr(chunk, "text", None) or getattr(chunk, "content", "")
                if text:
                    parts.append(text)
            return "".join(parts)
        except Exception as e:  # noqa: BLE001
            print(f"[langgraph 重试 {attempt}/3] {type(e).__name__}: {e}")
            time.sleep(2 * attempt)
    return ""  # 失败兜底 → 由 _should_continue 判定为结束


def agent_node(state: AgentState) -> dict:
    """agent 节点：让 LLM 决定「继续检索」还是「给出最终答案」。

    用 JSON 约束输出：要么 {tool, input}（继续检索），要么 {answer}（作答）。
    用 JSON 而非自由文本，把条件边判断变成确定性的字段解析，避免正则脆弱。
    """
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=TRAFFIC_SYSTEM_PROMPT)] + messages

    prompt = (
        "你是一名城市轨道交通运维 Agent。请决定下一步。\n"
        "可用检索工具：\n" + TOOL_DESCRIPTIONS + "\n"
        "规则：\n"
        "1. 若还需检索知识库才能回答，输出 JSON："
        '{"tool": "<工具名>", "input": "<检索语句>"}；\n'
        "2. 若已有足够信息回答，输出 JSON："
        '{"answer": "<最终答案>"}；\n'
        "3. 只输出一个 JSON 对象，不要输出任何其它内容。"
    )
    raw = _chat(messages + [HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=raw)]}


def tools_node(state: AgentState) -> dict:
    """tools 节点：解析 agent 节点的工具调用，执行纯检索，回写 ToolMessage。

    ToolMessage 经 add_messages 归约器关联回 AIMessage，形成
    「决策 → 执行 → 观察」闭环。不调 LLM，仅返回原文片段。
    """
    last = state["messages"][-1]
    raw = last.content if hasattr(last, "content") else str(last)

    tool_name, tool_input = None, ""
    data = _parse_json(raw)
    if isinstance(data, dict):
        tool_name = data.get("tool")
        tool_input = data.get("input", "")

    if tool_name is None or tool_name not in TOOL_NAMES:
        return {
            "messages": [
                ToolMessage(
                    content=f"错误：没有名为 '{tool_name}' 的工具，可用工具 {TOOL_NAMES}。",
                    tool_call_id="none",
                )
            ],
            "iterations": state.get("iterations", 0) + 1,
        }

    observation = retrieve(tool_input, category=tool_name, top_k=RETRIEVE_TOP_K)
    return {
        "messages": [
            ToolMessage(content=observation, name=tool_name, tool_call_id="none")
        ],
        "iterations": state.get("iterations", 0) + 1,
    }


def _last_tool_call(state: AgentState) -> Optional[dict]:
    """提取 agent 节点最新一条 AIMessage 中的工具调用（无则 None）。"""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    data = _parse_json(last.content)
    if isinstance(data, dict) and "tool" in data and "input" in data:
        return data
    return None


def _should_continue(state: AgentState) -> Literal["tools", "end"]:
    """条件边：有工具调用且未超迭代 → tools；否则 → end。"""
    if _last_tool_call(state) and state.get("iterations", 0) < MAX_TOOL_ITERATIONS:
        return "tools"
    return "end"


def _final_answer(state: AgentState) -> str:
    """从最终状态提取答案。

    优先取 agent 节点输出的 answer 字段；若模型未按 JSON 输出（偶发），
    回退到最后一条 AIMessage 的原文，避免「未生成答案」的报废式失败。
    """
    last_ai_text = None
    for m in reversed(state["messages"]):
        if not isinstance(m, AIMessage):
            continue
        if last_ai_text is None:
            last_ai_text = getattr(m, "content", "") or ""
        data = _parse_json(m.content)
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]
    # 回退：取最后一条非空 AI 输出原文（剥离 JSON 代码块围栏 / answer 字段包裹）
    if last_ai_text:
        t = last_ai_text.strip().strip("`").strip()
        # 若仍是 {"answer": "..."} 形态但 JSON 解析失败（内容含裸控制字符等），
        # 用正则剥离外层包裹，取出 answer 字段内的原文
        m = re.match(r'^\s*\{\s*"answer"\s*:\s*"(?P<body>[\s\S]*)"\s*\}\s*$', t)
        if m:
            t = m.group("body")
        return t or "（未生成答案，请稍后重试）"
    return "（未生成答案，请稍后重试）"


# ---------------------------------------------------------------------------
# 4. 组装状态图
# ---------------------------------------------------------------------------
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# 5. 对外接口（与 api/agent.py 的 TrafficAgent 对齐，方便无缝切换）
# ---------------------------------------------------------------------------
class LangGraphAgent:
    """LangGraph 版 ReAct Agent 的薄封装。

    与 api/agent.py.TrafficAgent 保持同构接口（run / chat），
    便于在 api/main.py 里通过开关切换新旧实现，做 A/B 对比。
    """

    def __init__(self, max_iter: int = MAX_TOOL_ITERATIONS):
        self.max_iter = max_iter
        self.graph = _compiled_graph

    def run(self, question: str, session_id: str = "default") -> dict:
        """一次问答：拼历史 → 跑图 → 提取答案 → 存历史。"""
        history = _store.get_history(session_id)
        history_msgs = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in history
        ]
        t0 = time.time()
        result = self.graph.invoke(
            {"messages": history_msgs + [HumanMessage(content=question)], "iterations": 0},
            config={"recursion_limit": 50},
        )
        answer = _final_answer(result)
        latency_ms = round((time.time() - t0) * 1000, 2)

        _store.append(session_id, "user", question)
        _store.append(session_id, "assistant", answer)
        return {"answer": answer, "latency_ms": latency_ms, "trace": result}

    def chat(self, question: str, session_id: str = "default") -> dict:
        return self.run(question, session_id=session_id)


def build_langgraph_agent() -> LangGraphAgent:
    """工厂：返回 LangGraph 版 Agent（供 api/main.py 切换用）。"""
    return LangGraphAgent()


if __name__ == "__main__":
    # 直接 `python -m api.langgraph_agent` 跑一次自测（真实调用 LLM + 检索）
    agent = LangGraphAgent()
    res = agent.run("信号系统维修分级与周期")
    print("答案：", res["answer"])
    print("耗时：", res["latency_ms"], "ms")
