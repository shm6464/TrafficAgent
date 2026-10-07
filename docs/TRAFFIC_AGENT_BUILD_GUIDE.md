# TrafficAgent 实现交接文档（自包含版）

> 本文档供另一位执行者照做，**不依赖会话上下文**。所有文件路径、函数签名、命令均已对照当前仓库核实。
> 目标：在现有 RAG 问答系统之上，增量加一层 **ReAct Agent 外壳 + 多轮记忆**，补齐「RAG → Agent」能力缺口。
> 硬约束（用户要求）：**只新增文件 / 新增端点，绝不改写 `api/service.py` 与 `libs/kotaemon/` 的检索/重排/缓存逻辑。**

---

## 0. 背景与目标

- **为什么做**：现有 `POST /query` 是单跳 RAG（一次检索 → 一次生成），答不好需要「先查分类、再查流程、再查降级」的复合问题，也答不了「那它的时限是几天？」这类指代追问。TrafficAgent 用 ReAct 循环让模型**多次调用 RAG 工具**、并用 `session_id` 维持多轮记忆。
- **为什么是增量**：仓库基于 kotaemon 改造（BM25+向量混合检索、bge 重排、双层缓存都在 `libs/kotaemon/` 里改过），重写会破坏已有工程化成果与 Apache-2.0 合规说明。
- **交付物**：`api/agent.py`（新建）+ `api/main.py` 新增 `POST /agent`（增改）。其余不动。

---

## 1. 当前仓库真实状态（已核实）

| 文件 | 状态 | 说明 |
|---|---|---|
| `api/agent.py` | **不存在** | 需新建（本文 §3.1 给出完整代码） |
| `api/main.py` | 存在，**无 `/agent`** | 仅有 `/`、`/docs`、`/healthz`、`/query`、`/ingest`；需新增 `AgentRequest` + `/agent` 路由（§3.2） |
| `api/service.py` | 存在，**勿改** | `TrafficQAService.query()` 是 RAG 工具来源 |
| `api/__init__.py` | 存在 | 确认包结构；若无则建空文件 |
| `libs/kotaemon/`、`libs/ktem/` | 存在，**勿改** | 检索/重排/缓存实现所在 |

### 关键接口（直接复用，不要重写）

```python
# api/service.py —— RAG 工具核心
from .service import get_service
svc = get_service()                       # 进程内单例
res = svc.query(question, top_k=5)        # -> dict
# res 关键字段：
#   res["answer"]      : str，生成答案
#   res["citations"]  : list[dict]，每项 {"doc","page","score","text"}
#   res["latency_ms"] : float

# LLM（DeepSeek 通道，由 flowsettings.py 的 OPENAI_* 配置）
from ktem.llms.manager import llms
llm = llms.get_default()                  # -> ChatOpenAI
out = llm.invoke([SystemMessage(content=...), HumanMessage(content=...)])
text = getattr(out, "text", None) or getattr(out, "content", "")

# 消息类型
from kotaemon.base import HumanMessage, SystemMessage

# （可选）框架自带 Agent/工具基类
from kotaemon.agents import ReactAgent, BaseTool
```

> 前置：`.env` 须含 `OPENAI_API_KEY` / `OPENAI_CHAT_MODEL`（走 DeepSeek）。否则运行期会 401，不是代码错。

---

## 2. 目标架构

```
用户问题
   │
   ▼
POST /agent  (api/main.py)  ──► AgentRequest{question, session_id, top_k}
   │
   ▼
TrafficAgent.chat(question, session_id)   # 拼历史 → run → 存历史
   │
   ▼
TrafficAgent.run(question)   ← 手写 ReAct 循环
   │   每轮：拼 REACT_TEMPLATE → llm.invoke → 解析 Action
   │   若 Action == knowledge_base → kb_search(input)
   ▼                                     ▲
kb_search(query, top_k)  ────────────────┘  返回「答案+参考来源」作为 Observation
   │
   ▼
TrafficQAService.query()  (api/service.py，复用既有 RAG 链路，不改)
```

- **多轮记忆**：`SESSIONS: dict[session_id -> [{"role","content"}]]` 进程内存储，把历史拼进 prompt。
- **单进程假设**：uvicorn 默认单进程，内存字典可用；多 worker 时需换外部存储（Redis），本阶段不要求。

---

## 3. 实现步骤（逐文件）

### 3.1 新建 `api/agent.py`（核心，完整代码）

直接新建该文件，内容如下（已修正此前 `chat` 放错位置、模板缩进、编码等坑）：

```python
# api/agent.py
from __future__ import annotations
import re
import time
import textwrap
from typing import Optional

from ktem.llms.manager import llms
from kotaemon.base import HumanMessage, SystemMessage
from .service import get_service

# ---- 多轮会话存储（进程内；单进程 uvicorn 下可用）----
SESSIONS: dict[str, list[dict]] = {}

def _history_block(session_id: str) -> str:
    hist = SESSIONS.get(session_id, [])
    if not hist:
        return ""
    lines = [f"{'用户' if m['role']=='user' else '助手'}: {m['content']}" for m in hist]
    return "历史对话：\n" + "\n".join(lines) + "\n\n"

# ---- RAG 工具：返回给 Agent 作为 Observation 的可读文本 ----
def kb_search(query: str, top_k: int = 5) -> str:
    svc = get_service()
    res = svc.query(query, top_k=top_k)
    answer = res.get("answer", "")
    cites = res.get("citations", [])
    tail = "\n".join(
        f"[{i+1}] {c.get('doc','-')}（页码 {c.get('page')}）"
        for i, c in enumerate(cites[:3])
    )
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

class TrafficAgent:
    def __init__(self, max_iter: int = 5, top_k: int = 5):
        self.llm = llms.get_default()
        self.top_k = top_k
        self.max_iter = max_iter
        self.tool_name = "knowledge_base"
        self.tool_desc = (
            "knowledge_base：城市轨道交通运维知识库检索工具。"
            "当你需要规章条款、处置流程、设备参数等内部知识时调用，"
            "输入应为尽量具体的检索语句。"
        )

    def _call_llm(self, prompt: str) -> str:
        resp = self.llm.invoke([
            SystemMessage(content="你是严谨的运维 Agent，按 ReAct 格式输出。"),
            HumanMessage(content=prompt),
        ])
        return getattr(resp, "text", None) or getattr(resp, "content", "")

    def _parse_action(self, text: str):
        """解析 LLM 输出：Final Answer / Action+Input / 退化兜底。"""
        if "Final Answer:" in text:
            return ("final", text.split("Final Answer:", 1)[1].strip())
        m = re.search(r"Action:\s*([^\n]+)", text)
        if not m:
            return ("final", text.strip())          # 模型没按格式 → 当最终答案
        name = m.group(1).strip()
        inp = ""
        mi = re.search(r"Action Input:\s*(.*?)(?=\nObservation:|\Z)", text, re.S)
        if mi:
            inp = mi.group(1).strip().strip('"').strip("'")
        return ("action", name, inp)

    def run(self, question: str) -> dict:
        scratch = ""
        t0 = time.time()
        answer = ""
        for _ in range(self.max_iter):
            prompt = REACT_TEMPLATE.format(
                tool_description=self.tool_desc,
                tool_names=self.tool_name,
                instruction=question,
                agent_scratchpad=scratch,
            )
            out = self._call_llm(prompt)
            kind = self._parse_action(out)
            if kind[0] == "final":
                answer = kind[1]
                break
            _, name, inp = kind
            if name != self.tool_name:
                obs = f"错误：没有名为 '{name}' 的工具，可用工具只有 [{self.tool_name}]。"
            else:
                obs = kb_search(inp, top_k=self.top_k)
            scratch += f"\n{out}\nObservation: {obs}\n"
        else:
            # 达到 max_iter 仍未 Final Answer：用最后一次检索兜底
            answer = kb_search(question, top_k=self.top_k)

        return {
            "answer": answer,
            "trace": scratch,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }

    def chat(self, question: str, session_id: str = "default") -> dict:
        """多轮会话入口：拼历史 -> run -> 存历史。供 api/main.py 的 /agent 调用。"""
        pre = _history_block(session_id)
        full_q = f"{pre}Question: {question}" if pre else question
        res = self.run(full_q)
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": question})
        SESSIONS[session_id].append({"role": "assistant", "content": res["answer"]})
        return res

if __name__ == "__main__":
    # 直接 `python -m api.agent` 跑一次自测（会真实调用一次 DeepSeek）
    print(kb_search("信号系统维修分级与周期", top_k=3))
```

> **关键正确性点（此前踩过的坑，务必保留）**：`chat` 是 `TrafficAgent` 的**类方法**（4 空格缩进在 `class` 内）。写成模块级函数会导致 `POST /agent` 返回 500 `AttributeError`。

### 3.2 修改 `api/main.py`（新增 `AgentRequest` + `/agent` 路由）

在 `IngestRequest` 类定义之后（约第 92 行后）插入请求模型：

```python
class AgentRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: str = Field("default", description="多轮会话 id")
    top_k: int = Field(5, ge=1, le=20, description="检索文档数")
```

在 `ingest` 路由函数之后（约第 198 行后）插入路由：

```python
@app.post("/agent")
def agent_chat(req: AgentRequest):
    """ReAct Agent 问答：把 RAG 包成可多步检索 + 多轮记忆的 Agent。"""
    from .agent import TrafficAgent

    agent = TrafficAgent(top_k=req.top_k)
    res = agent.chat(req.question, session_id=req.session_id)
    return {
        "answer": res["answer"],
        "latency_ms": res["latency_ms"],
        "session_id": req.session_id,
    }
```

（可选）同步更新文件顶部 docstring 与 `root()` 的 HTML，加入 `/agent` 说明，保持文档一致。

---

## 4. 验证（必做）

> 全程使用项目 `.venv` 的解释器；命令在 **PowerShell** 中执行。

### Step 1 — 单元自测（确认 RAG 工具能返回文本）
```powershell
cd D:/google/kotaemon-main/kotaemon-main
.venv\Scripts\python.exe -m api.agent
```
预期：打印一段带「参考来源」的 KB 答案（会真实调用一次 DeepSeek）。
> 注意用 `-m api.agent` 而非 `python api/agent.py`，否则相对导入报错。

### Step 2 — 启动服务（双终端）
- **终端 A**（常驻）：
```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
（`--reload` 让改代码后自动重启；不加则每次改动需手动 `Ctrl+C` 重启。）
- **终端 B**：执行下面的测试命令。

### Step 3 — 测试 `/agent`
```powershell
$json = '{"question":"信号系统维修分级与周期","session_id":"s1"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/agent -Method Post -ContentType "application/json" -Body $json
```
预期：打印出 `answer` 内容，无 `detail` 报错 → **Agent 端到端跑通**。

> 调试时想看 ReAct 的 Thought/Action/Observation 循环：在 `TrafficAgent.run` 返回里已有 `trace` 字段，可在 `/agent` 路由临时把 `res["trace"]` 也返回给前端查看。

---

## 5. 必看避坑清单（照做，别重踩）

| # | 坑 | 现象 | 正确做法 |
|---|---|---|---|
| 1 | 用错 Python | `ModuleNotFoundError: No module named 'langchain'` | 永远用 `.venv\Scripts\python.exe`（langchain 只装在 .venv） |
| 2 | 直接 `python api/agent.py` | `ImportError: attempted relative import with no known parent package` | 改成 `python -m api.agent` |
| 3 | 改代码后没重启 uvicorn | 改动不生效 / 仍 500 | 加 `--reload`，或手动 `Ctrl+C` 重启 |
| 4 | PowerShell 里用 `curl` | `Invoke-WebRequest` 参数类型错 | 用 `curl.exe` 或 `Invoke-RestMethod` |
| 5 | `curl.exe -d '{"question":"中文"}'` | FastAPI 422 `There was an error parsing the body` | 中文被 GBK 编码；改用 `Invoke-RestMethod`（UTF-8）或 `.py` 文件或 `python urllib` 显式 utf-8 |
| 6 | `python -c '...'` 带双引号 | `SyntaxError`（PowerShell 剥离双引号） | 别用 `-c`，落地成 `.py` 文件或用 `Invoke-RestMethod` |
| 7 | `chat` 写成模块级函数 | `POST /agent` 500 `AttributeError` | `chat` 必须是 `TrafficAgent` 的**类方法** |

---

## 6. 进阶（可选，非必须）

可用 kotaemon 自带的 `ReactAgent` 把 RAG 注册成 `BaseTool`，由框架接管循环（生产化更省心）：
1. 新建 `api/kb_tool.py`，继承 `kotaemon.agents.tools.BaseTool`，在 `_run` 里调 `kb_search`；
2. `agent = ReactAgent.withx(plugins=[KBTool()])`（具体 API 见 `libs/kotaemon/kotaemon/agents/react/agent.py`）；
3. 复用 §3.1 的 `REACT_TEMPLATE` 思路或直接用框架默认 `zero_shot_react_prompt`（`kotaemon/agents/react/prompt.py`）。

> 手写版（§3.1）已实现完整 ReAct 循环，足够交付；框架版作为后续增强。

---

## 7. 简历 / 叙事对齐

- 对外表述统一为：**「基于 kotaemon 改造的城市轨道运维知识库；在 RAG 之上增量实现了 ReAct Agent（多步检索 + 多轮记忆），补齐单跳 RAG 答不好复合问题的短板」**。
- 与 TradingAgents 项目形成「单 Agent + 多 Agent」完整叙事；强调「只做增量增强，未重写框架」。
- 评测口径统一：Recall@5 / 忠实度等数字须有可复现脚本支撑（见既有 `eval/` 与 `docs/` 报告）。

---

## 8. 验收标准（Definition of Done）

- [ ] `api/agent.py` 新建，`TrafficAgent.chat` 为类方法，模块自测 `python -m api.agent` 通过；
- [ ] `api/main.py` 新增 `AgentRequest` + `POST /agent`，服务启动无报错；
- [ ] `Invoke-RestMethod` 测 `/agent` 返回含 `answer` 的 JSON，无 500/422；
- [ ] 多轮：同一 `session_id` 连续提问，能基于上文回答指代问题；
- [ ] `service.py` 与 `libs/kotaemon/` **未被修改**（增量约束成立）；
- [ ] 能展示一次 ReAct trace（Thought→Action→Observation→Final Answer 至少一轮）。
