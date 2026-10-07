# api/agent.py
from __future__ import annotations

import json
import re
import time
import textwrap

from ktem.llms.manager import llms
from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT, TRAFFIC_SYSTEM_PROMPT
from kotaemon.base import HumanMessage, SystemMessage

from .service import get_service
from .memory_store import get_memory_store

# ---- 多轮会话存储（可插拔：默认内存，MEMORY_BACKEND=redis 可切换）----
_store = get_memory_store()


def _history_block(session_id: str) -> str:
    hist = _store.get_history(session_id)
    if not hist:
        return ""
    lines = [f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in hist]
    return "历史对话：\n" + "\n".join(lines) + "\n\n"


# ---- 多工具定义：按文档类别划分的知识库检索工具 ----
TOOLS = {
    "search_regulations": {
        "desc": "检索规章规范类知识（管理办法、技术规范、行车组织规定）",
        "keywords": ("管理办法", "技术规范", "行车组织"),
    },
    "search_equipment": {
        "desc": "检索系统设备类知识（CBTC、牵引供电、站台门、AFC 等设备原理与参数）",
        "keywords": ("原理", "监控", "站台门", "屏蔽门", "对比", "定位", "测速", "再生制动", "售检票", "AFC", "概览"),
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


def _classify_file(file_name: str) -> str:
    """按文件名关键词把文档归到某个工具类别（应急 > 规章 > 检修 > 设备默认）。"""
    if any(k in file_name for k in ("应急", "火灾", "预案")):
        return "search_emergency"
    if any(k in file_name for k in ("管理办法", "技术规范", "行车组织")):
        return "search_regulations"
    if any(k in file_name for k in ("检修", "修程", "维修分级", "养护")):
        return "search_maintenance"
    return "search_equipment"


# ---- RAG 工具：检索 + 按类别过滤 + 生成，返回给 Agent 作为 Observation ----
def kb_search(query: str, category: str = None, top_k: int = 5) -> str:
    svc = get_service()
    # 拉大检索量，再按类别过滤（不改 libs/kotaemon，仅在 agent 层做文档分类）
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
    messages = [SystemMessage(content=TRAFFIC_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    # 重试容错（DeepSeek 高峰时段偶发超时，与 service.query 同款策略）
    resp = None
    for attempt in range(1, 4):
        try:
            resp = svc.llm.invoke_stream(messages)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[kb_search 重试 {attempt}/3] {type(e).__name__}: {e}")
            time.sleep(2 * attempt)
    if resp is None:
        answer = "（检索生成失败，请稍后重试）"
    else:
        answer = getattr(resp, "text", None) or getattr(resp, "content", "")
    cites = [d.metadata.get("file_name", "-") for d in docs[:3]]
    tail = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(cites))
    return f"{answer}\n\n参考来源：\n{tail}" if tail else answer


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


class TrafficAgent:
    def __init__(self, max_iter: int = 5, top_k: int = 5):
        self.llm = llms.get_default()
        self.top_k = top_k
        self.max_iter = max_iter
        self.tool_names = list(TOOLS.keys())
        self.tool_desc = "\n".join(
            f"- {name}：{info['desc']}" for name, info in TOOLS.items()
        )

    def _stream_llm(self, prompt: str):
        """token 级流式调用 LLM（带重试）。逐 token yield，供前端实时展示推理过程。"""
        messages = [
            SystemMessage(content="你是严谨的运维 Agent，按 ReAct 格式输出。"),
            HumanMessage(content=prompt),
        ]
        for attempt in range(1, 4):
            try:
                for chunk in self.llm.stream(messages):
                    token = getattr(chunk, "text", None) or getattr(chunk, "content", "")
                    if token:
                        yield token
                return
            except Exception as e:  # noqa: BLE001
                print(f"[ReAct 重试 {attempt}/3] {type(e).__name__}: {e}")
                time.sleep(2 * attempt)
        # 三次重试均失败：不 yield 任何 token，调用方按 final 兜底

    def _stream_final(self, prompt: str):
        """token 级流式生成最终答案（逐 token yield）。"""
        for chunk in self.llm.stream([
            SystemMessage(content="你是严谨的运维 Agent，请给出准确、结构清晰的回答。"),
            HumanMessage(content=prompt),
        ]):
            token = getattr(chunk, "text", None) or getattr(chunk, "content", "")
            if token:
                yield token

    def _parse_action(self, text: str):
        """解析 LLM 输出：优先执行 Action；无 Action 才看 Final Answer。

        关键：DeepSeek 常一次性脑补完整 ReAct 循环（含伪造的 Observation 与
        Final Answer）。因此必须**优先取 Action**，把真实工具结果注入后再继续，
        而不是直接采信模型自编的 Final Answer。
        """
        # 1) 先找 Action —— 有 Action 就执行它（忽略其后的 Observation/Final Answer）
        m = re.search(r"Action:\s*([^\n]+)", text)
        if m:
            name = m.group(1).strip()
            inp = ""
            mi = re.search(
                r"Action Input:\s*(.*?)(?=\nObservation:|\nThought:|\nFinal Answer:|\nAction:|\Z)",
                text,
                re.S,
            )
            if mi:
                inp = mi.group(1).strip().strip('"').strip("'")
            return ("action", name, inp)
        # 2) 无 Action → Final Answer
        if "Final Answer:" in text:
            return ("final", text.split("Final Answer:", 1)[1].strip())
        # 3) 退化兜底：模型完全没按格式 → 当最终答案
        return ("final", text.strip())

    @staticmethod
    def _strip_after_observation(text: str) -> str:
        """截断到 Observation / Final Answer 之前，丢弃 LLM 脑补的后续内容。"""
        for marker in ("Observation:", "Final Answer:"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].rstrip()
        return text

    @staticmethod
    def _extract_thought(text: str) -> str:
        """从 LLM 输出里提取 Thought 内容（Thought: 到 Action:/Final Answer: 之间）。"""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|\Z)", text, re.S)
        if m:
            return m.group(1).strip()
        return ""

    def _react_steps(self, question: str, final_question: str = None):
        """核心 ReAct 循环（生成器）：工具收集 + token 级流式答案生成。

        事件类型：
          thought_token —— LLM 推理过程的 token（逐字流式，前端实时展示）
          thought      —— LLM 的完整思考（兜底/未走流式时使用）
          action       —— 要执行的动作（含 name/input）
          status       —— 过程状态提示
          observation  —— 工具返回的结果
          final_token  —— 最终答案的 token（逐字流式）
          done         —— 结束（含 trace/latency_ms/answer）
        """
        scratch = ""
        observations = []
        t0 = time.time()
        # 阶段1：工具循环（流式输出推理 token，再解析 Action）
        for i in range(self.max_iter):
            yield {"type": "status", "content": "Agent 正在分析问题…" if i == 0 else "Agent 正在进一步推理…"}
            prompt = REACT_TEMPLATE.format(
                tool_description=self.tool_desc,
                tool_names=", ".join(self.tool_names),
                instruction=question,
                agent_scratchpad=scratch,
            )
            # 流式输出推理过程：逐 token 发给前端，消除"思考卡顿"的静止感
            out_parts = []
            for token in self._stream_llm(prompt):
                out_parts.append(token)
                yield {"type": "thought_token", "content": token}
            out = "".join(out_parts)
            if not out.strip():
                break  # LLM 调用失败兜底：直接进入答案生成
            kind = self._parse_action(out)
            if kind[0] == "final":
                break          # 信息收集完成，进入流式答案生成
            _, name, inp = kind
            if not inp.strip():
                inp = question          # Action Input 为空时退回原问题检索
            thought = self._extract_thought(out)
            if thought:
                yield {"type": "thought", "content": thought}
            yield {"type": "action", "name": name, "input": inp}
            if name not in self.tool_names:
                obs = f"错误：没有名为 '{name}' 的工具，可用工具只有 [{', '.join(self.tool_names)}]。"
            else:
                yield {"type": "status", "content": f"正在检索知识库并综合分析（{name}）：{inp}"}
                obs = kb_search(inp, category=name, top_k=self.top_k)
            observations.append(obs)
            yield {"type": "observation", "content": obs}
            # 只保留到 Action Input 为止，丢弃 LLM 脑补的 Observation/Final Answer
            head = self._strip_after_observation(out)
            scratch += f"\n{head}\nObservation: {obs}\n"

        # 兜底：一轮都没检索到（LLM 直接 final / 异常），先补一次检索
        if not observations:
            yield {"type": "status", "content": "正在检索知识库…"}
            obs = kb_search(question, top_k=self.top_k)
            observations.append(obs)
            yield {"type": "observation", "content": obs}

        # 阶段2：token 级流式答案生成
        evidence = "\n\n".join(
            f"【检索结果 {i+1}】\n{o}" for i, o in enumerate(observations)
        )
        final_q = final_question or question
        final_prompt = FINAL_TEMPLATE.format(evidence=evidence, question=final_q)
        yield {"type": "status", "content": "正在生成答案…"}
        answer = ""
        for token in self._stream_final(final_prompt):
            answer += token
            yield {"type": "final_token", "content": token}

        yield {
            "type": "done",
            "trace": scratch,
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "answer": answer,
        }

    def run(self, question: str, final_question: str = None) -> dict:
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
        """流式会话入口：拼历史 -> 逐步骤 yield SSE -> 存历史。供 /agent/stream 调用。"""
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
        """多轮会话入口：拼历史 -> run -> 存历史。供 api/main.py 的 /agent 调用。"""
        pre = _history_block(session_id)
        full_q = f"{pre}Question: {question}" if pre else question
        res = self.run(full_q, final_question=question)
        _store.append(session_id, "user", question)
        _store.append(session_id, "assistant", res["answer"])
        return res


if __name__ == "__main__":
    # 直接 `python -m api.agent` 跑一次自测（会真实调用一次 DeepSeek）
    print(kb_search("信号系统维修分级与周期", top_k=3))
