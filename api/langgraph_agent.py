# api/langgraph_agent.py
"""LangGraph 版 ReAct Agent（统一入口）。

用 LangGraph StateGraph 实现「思考 → 检索 → 观察 → 再思考 → 作答」的 ReAct 循环，
同时支持 token 级流式推理（SSE 推送 thought_token / action / observation / final_token）。

图结构：
    START → agent（LLM 流式推理）──条件边──┬─ 需检索 → tools（纯检索）
           ▲                                │              │
           └────────────────────────────────┴──────────────┘
                 Final Answer / 超迭代 → END

设计要点：
    - agent_node 使用 ReAct 格式 prompt（Thought/Action/Action Input/Final Answer），
      通过正则解析 LLM 输出，决定路由到 tools 还是 END；
    - tools_node 只做纯检索（不调 LLM），按类别过滤文档，返回原文片段；
    - 流式路径（stream_chat）手动驱动图节点执行，逐 token yield SSE 事件；
    - 非流式路径（run）同样驱动图节点，但不 yield 中间事件；
    - 本文件自包含工具定义和 ReAct 逻辑，不依赖 api/agent.py。
"""
from __future__ import annotations

import json
import re
import time
import textwrap
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage as LCHumanMessage
from langchain_core.messages import SystemMessage as LCSystemMessage
from langchain_core.messages import ToolMessage as LCToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from kotaemon.base import HumanMessage as KHumanMessage
from kotaemon.base import SystemMessage as KSystemMessage

from ktem.llms.manager import llms
from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT, TRAFFIC_SYSTEM_PROMPT

from .service import get_service
from .memory_store import get_memory_store

_store = get_memory_store()

MAX_TOOL_ITERATIONS = 5
RETRIEVE_TOP_K = 5


# ---------------------------------------------------------------------------
# 1. 工具定义 & 文档分类
# ---------------------------------------------------------------------------
TOOLS = {
    "search_regulations": {
        "desc": "检索规章规范类知识（管理办法、技术规范、行车组织规定）",
        "keywords": ("管理办法", "技术规范", "行车组织"),
    },
    "search_equipment": {
        "desc": "检索系统设备类知识（CBTC、牵引供电、站台门、AFC 等设备原理与参数）",
        "keywords": ("原理", "监控", "站台门", "屏蔽门", "对比", "定位", "测速",
                     "再生制动", "售检票", "AFC", "概览"),
    },
    "search_maintenance": {
        "desc": "检索检修维护类知识（检修修程、维修分级、养护周期）",
        "keywords": ("检修", "修程", "维修分级", "养护"),
    },
    "search_emergency": {
        "desc": "检索应急处突类知识（突发事件应急预案、火灾处置、预案体系）",
        "keywords": ("应急", "火灾", "预案"),
    },
}

TOOL_NAMES = list(TOOLS.keys())
TOOL_DESCRIPTIONS = "\n".join(
    f"- {name}：{info['desc']}" for name, info in TOOLS.items()
)


def _classify_file(file_name: str) -> str:
    """按文件名关键词把文档归到某个工具类别（应急 > 规章 > 检修 > 设备默认）。"""
    if any(k in file_name for k in ("应急", "火灾", "预案")):
        return "search_emergency"
    if any(k in file_name for k in ("管理办法", "技术规范", "行车组织")):
        return "search_regulations"
    if any(k in file_name for k in ("检修", "修程", "维修分级", "养护")):
        return "search_maintenance"
    return "search_equipment"


# ---------------------------------------------------------------------------
# 2. 检索工具（纯检索，不调 LLM）
# ---------------------------------------------------------------------------
def retrieve(query: str, category: str = None, top_k: int = RETRIEVE_TOP_K) -> str:
    """执行一次纯检索，返回原文片段 + 出处。"""
    svc = get_service()
    docs = svc.retriever.run(query, top_k=max(top_k * 4, 20))
    if category:
        docs = [d for d in docs if _classify_file(d.metadata.get("file_name", "")) == category]
    if not docs:
        return "未在知识库中检索到相关内容。"
    return "\n\n".join(
        f"[出处{i}: {d.metadata.get('file_name', '-')}]\n{d.text}"
        for i, d in enumerate(docs[:top_k], 1)
    )


def kb_search(query: str, category: str = None, top_k: int = RETRIEVE_TOP_K) -> str:
    """检索 + LLM 摘要生成，返回带参考来源的 Observation 文本。"""
    try:
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
        prompt = TRAFFIC_QA_TEXT_PROMPT.format(lang="中文", context=evidence, question=query)
        messages = [KSystemMessage(content=TRAFFIC_SYSTEM_PROMPT), KHumanMessage(content=prompt)]
        resp = None
        for attempt in range(1, 4):
            try:
                resp = svc.llm.invoke_stream(messages)
                break
            except Exception as e:
                print(f"[kb_search 重试 {attempt}/3] {type(e).__name__}: {e}")
                time.sleep(2 * attempt)
        if resp is None:
            answer = "（检索生成失败，请稍后重试）"
        else:
            answer = getattr(resp, "text", None) or getattr(resp, "content", "")
        cites = [d.metadata.get("file_name", "-") for d in docs[:3]]
        tail = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(cites))
        return f"{answer}\n\n参考来源：\n{tail}" if tail else answer
    except Exception as e:
        print(f"[kb_search 错误] {type(e).__name__}: {e}")
        return f"检索失败：{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 3. Prompt 模板
# ---------------------------------------------------------------------------
REACT_TEMPLATE = textwrap.dedent("""\
你是一名「城市轨道交通运维 Agent」，服务于一线运维与调度人员。
请严格基于【工具】返回的内容作答；若工具未检索到相关内容，必须明确说明"未在知识库中检索到"，严禁编造。

你拥有以下工具：
{tool_description}

使用以下格式推理：
Question: 必须回答的输入问题
Thought: 你应当思考下一步要做什么
Action: 要执行的动作，必须是 [{tool_names}] 之一
Action Input: 动作的输入（尽量具体的检索语句）
Observation: 动作返回的结果
...（Thought/Action/Action Input/Observation 可重复多次）
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

开始！
Question: {instruction}
Thought:{agent_scratchpad}""")

FINAL_TEMPLATE = textwrap.dedent("""\
你是一名「城市轨道交通运维 Agent」。请基于以下【检索结果】回答用户问题。

要求：
1. 严格依据检索结果作答，严禁编造；
2. 若检索结果不足以回答，明确说明「未在知识库中检索到」；
3. 回答结构清晰，可适当使用列表或表格。

【检索结果】
{evidence}

【用户问题】
{question}

请回答：""")


# ---------------------------------------------------------------------------
# 4. LLM 流式调用 & 解析
# ---------------------------------------------------------------------------
def _stream_llm(prompt: str):
    """token 级流式 LLM 调用（带 3 次重试）。"""
    messages = [
        KSystemMessage(content="你是严谨的运维 Agent，按 ReAct 格式输出。"),
        KHumanMessage(content=prompt),
    ]
    llm_inst = llms.get_default()
    for attempt in range(1, 4):
        try:
            for chunk in llm_inst.stream(messages):
                token = getattr(chunk, "text", None) or getattr(chunk, "content", "")
                if token:
                    yield token
            return
        except Exception as e:
            print(f"[ReAct 重试 {attempt}/3] {type(e).__name__}: {e}")
            time.sleep(2 * attempt)


def _stream_final_llm(prompt: str):
    """token 级流式生成最终答案。"""
    messages = [
        KSystemMessage(content="你是严谨的运维 Agent，请给出准确、结构清晰的回答。"),
        KHumanMessage(content=prompt),
    ]
    for chunk in llms.get_default().stream(messages):
        token = getattr(chunk, "text", None) or getattr(chunk, "content", "")
        if token:
            yield token


def _parse_action(text: str):
    """解析 LLM 输出：优先取 Action（防 LLM 脑补完整 ReAct 链），无 Action 才看 Final Answer。"""
    m = re.search(r"Action:\s*([^\n]+)", text)
    if m:
        name = m.group(1).strip()
        inp = ""
        mi = re.search(
            r"Action Input:\s*(.*?)(?=\nObservation:|\nThought:|\nFinal Answer:|\nAction:|\Z)",
            text, re.S,
        )
        if mi:
            inp = mi.group(1).strip().strip('"').strip("'")
        return ("action", name, inp)
    if "Final Answer:" in text:
        return ("final", text.split("Final Answer:", 1)[1].strip())
    return ("final", text.strip())


def _extract_thought(text: str) -> str:
    """提取 Thought 内容。"""
    m = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def _strip_after_observation(text: str) -> str:
    """截断到 Observation / Final Answer 之前，丢弃 LLM 脑补的后续内容。"""
    for marker in ("Observation:", "Final Answer:"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
    return text


# ---------------------------------------------------------------------------
# 5. LangGraph 状态 & 节点
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """LangGraph 共享状态。"""
    messages: Annotated[list, add_messages]
    iterations: int


def _to_kotaemon_message(msg):
    """langchain_core 消息 → kotaemon 消息类型转换。"""
    content = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(msg, LCSystemMessage):
        return KSystemMessage(content=content)
    if isinstance(msg, LCToolMessage):
        return KHumanMessage(content=f"[检索结果]\n{content}")
    if isinstance(msg, LCAIMessage):
        return KHumanMessage(content=content)
    return KHumanMessage(content=content)


def agent_node(state: AgentState) -> dict:
    """agent 节点：LLM 流式推理，输出 ReAct 格式（Thought/Action/Final Answer）。"""
    scratch = ""
    for m in state["messages"]:
        if isinstance(m, LCToolMessage):
            scratch += f"\nObservation: {m.content}\n"
        elif isinstance(m, LCAIMessage):
            scratch += f"\n{_strip_after_observation(m.content)}\n"

    prompt = REACT_TEMPLATE.format(
        tool_description=TOOL_DESCRIPTIONS,
        tool_names=", ".join(TOOL_NAMES),
        instruction=state["messages"][0].content if state["messages"] else "",
        agent_scratchpad=scratch,
    )
    parts = []
    for token in _stream_llm(prompt):
        parts.append(token)
    out = "".join(parts)
    return {"messages": [LCAIMessage(content=out)]}


def tools_node(state: AgentState) -> dict:
    """tools 节点：解析 Action，执行检索，返回 ToolMessage。"""
    last = state["messages"][-1]
    out = last.content if hasattr(last, "content") else str(last)

    kind = _parse_action(out)
    if kind[0] != "action":
        return {
            "messages": [LCToolMessage(content="解析失败", tool_call_id="none")],
            "iterations": state.get("iterations", 0) + 1,
        }

    _, name, inp = kind
    if not inp.strip():
        first_human = state["messages"][0]
        inp = first_human.content if hasattr(first_human, "content") else str(first_human)

    if name not in TOOL_NAMES:
        obs = f"错误：没有名为 '{name}' 的工具，可用工具 {TOOL_NAMES}。"
    else:
        obs = kb_search(inp, category=name, top_k=RETRIEVE_TOP_K)

    return {
        "messages": [LCToolMessage(content=obs, name=name, tool_call_id="none")],
        "iterations": state.get("iterations", 0) + 1,
    }


def _should_continue(state: AgentState) -> Literal["tools", "end"]:
    """条件边：有 Action 且未超迭代 → tools，否则 → end。"""
    last = state["messages"][-1]
    if not isinstance(last, LCAIMessage):
        return "end"
    kind = _parse_action(last.content)
    if kind[0] == "action" and state.get("iterations", 0) < MAX_TOOL_ITERATIONS:
        return "tools"
    return "end"


def _final_answer_from_state(state: AgentState) -> str:
    """从最终状态提取答案。"""
    for m in reversed(state["messages"]):
        if isinstance(m, LCAIMessage):
            text = m.content.strip()
            if "Final Answer:" in text:
                return text.split("Final Answer:", 1)[1].strip()
            if text:
                return text
    return "（未生成答案，请稍后重试）"


# ---------------------------------------------------------------------------
# 6. 组装状态图
# ---------------------------------------------------------------------------
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# 7. 对外接口
# ---------------------------------------------------------------------------
def _history_block(session_id: str) -> str:
    hist = _store.get_history(session_id)
    if not hist:
        return ""
    lines = [f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in hist]
    return "历史对话：\n" + "\n".join(lines) + "\n\n"


class LangGraphAgent:
    """LangGraph 版 ReAct Agent，支持流式和非流式两种模式。"""

    def __init__(self, max_iter: int = MAX_TOOL_ITERATIONS, top_k: int = RETRIEVE_TOP_K):
        self.max_iter = max_iter
        self.top_k = top_k
        self.graph = _compiled_graph

    def _react_steps(self, question: str, final_question: str = None):
        """核心 ReAct 循环（生成器）：工具收集 + token 级流式答案生成。

        事件类型：
          thought_token —— LLM 推理过程逐 token
          thought      —— LLM 的完整思考
          action       —— 要执行的动作
          status       —— 过程状态提示
          observation  —— 工具返回的结果
          final_token  —— 最终答案逐 token
          done         —— 结束（含 trace/latency_ms/answer）
        """
        scratch = ""
        observations = []
        t0 = time.time()

        try:
          for i in range(self.max_iter):
            yield {"type": "status", "content": "Agent 正在分析问题…" if i == 0 else "Agent 正在进一步推理…"}
            prompt = REACT_TEMPLATE.format(
                tool_description=TOOL_DESCRIPTIONS,
                tool_names=", ".join(TOOL_NAMES),
                instruction=question,
                agent_scratchpad=scratch,
            )
            out_parts = []
            try:
                for token in _stream_llm(prompt):
                    out_parts.append(token)
                    yield {"type": "thought_token", "content": token}
            except Exception as e:
                print(f"[ReAct LLM 流式调用错误] {type(e).__name__}: {e}")
                yield {"type": "status", "content": f"LLM 调用出错：{type(e).__name__}，正在尝试继续…"}
                break
            out = "".join(out_parts)
            if not out.strip():
                break

            kind = _parse_action(out)
            if kind[0] == "final":
                break

            _, name, inp = kind
            if not inp.strip():
                inp = question

            thought = _extract_thought(out)
            if thought:
                yield {"type": "thought", "content": thought}
            yield {"type": "action", "name": name, "input": inp}

            if name not in TOOL_NAMES:
                obs = f"错误：没有名为 '{name}' 的工具，可用工具 [{', '.join(TOOL_NAMES)}]。"
            else:
                yield {"type": "status", "content": f"正在检索知识库并综合分析（{name}）：{inp}"}
                try:
                    obs = kb_search(inp, category=name, top_k=self.top_k)
                except Exception as e:
                    print(f"[kb_search 未捕获异常] {type(e).__name__}: {e}")
                    obs = f"检索出错：{type(e).__name__}: {e}"

            observations.append(obs)
            yield {"type": "observation", "content": obs}
            head = _strip_after_observation(out)
            scratch += f"\n{head}\nObservation: {obs}\n"

          if not observations:
            yield {"type": "status", "content": "正在检索知识库…"}
            try:
                obs = kb_search(question, top_k=self.top_k)
            except Exception as e:
                print(f"[kb_search 未捕获异常] {type(e).__name__}: {e}")
                obs = f"检索出错：{type(e).__name__}: {e}"
            observations.append(obs)
            yield {"type": "observation", "content": obs}

          evidence = "\n\n".join(
              f"【检索结果 {i+1}】\n{o}" for i, o in enumerate(observations)
          )
          final_q = final_question or question
          final_prompt = FINAL_TEMPLATE.format(evidence=evidence, question=final_q)
          yield {"type": "status", "content": "正在生成答案…"}
          answer = ""
          try:
              for token in _stream_final_llm(final_prompt):
                  answer += token
                  yield {"type": "final_token", "content": token}
          except Exception as e:
              print(f"[最终答案 LLM 流式调用错误] {type(e).__name__}: {e}")
              answer = f"生成答案时出错：{type(e).__name__}: {e}"
              yield {"type": "final_token", "content": answer}

          yield {
              "type": "done",
              "trace": scratch,
              "latency_ms": round((time.time() - t0) * 1000, 2),
              "answer": answer,
          }
        except Exception as e:
            print(f"[Agent 未捕获异常] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "type": "done",
                "trace": scratch,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "answer": f"Agent 执行出错：{type(e).__name__}: {e}",
            }

    def run(self, question: str, final_question: str = None) -> dict:
        """非流式执行 ReAct 循环，返回最终结果。"""
        answer = ""
        trace = ""
        latency = 0
        for step in self._react_steps(question, final_question=final_question):
            if step["type"] == "done":
                answer = step.get("answer", "")
                trace = step["trace"]
                latency = step["latency_ms"]
        return {"answer": answer, "trace": trace, "latency_ms": latency}

    def stream_chat(self, question: str, session_id: str = "default"):
        """流式会话入口：拼历史 → 逐步骤 yield SSE → 存历史。"""
        pre = _history_block(session_id)
        full_q = f"{pre}Question: {question}" if pre else question
        answer = ""
        for step in self._react_steps(full_q, final_question=question):
            if step["type"] == "done":
                answer = step.get("answer", "")
            yield f"data: {json.dumps(step, ensure_ascii=False)}\n\n"
        _store.append(session_id, "user", question)
        _store.append(session_id, "assistant", answer)

    def chat(self, question: str, session_id: str = "default") -> dict:
        """非流式会话入口：拼历史 → run → 存历史。"""
        pre = _history_block(session_id)
        full_q = f"{pre}Question: {question}" if pre else question
        res = self.run(full_q, final_question=question)
        _store.append(session_id, "user", question)
        _store.append(session_id, "assistant", res["answer"])
        return res


def build_langgraph_agent() -> LangGraphAgent:
    """工厂函数。"""
    return LangGraphAgent()


if __name__ == "__main__":
    agent = LangGraphAgent()
    res = agent.run("信号系统维修分级与周期")
    print("答案：", res["answer"])
    print("耗时：", res["latency_ms"], "ms")
