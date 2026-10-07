# 06_langgraph_react_report.md — LangGraph 重写 ReAct 循环

> 目标：把 `api/agent.py` 手写的 while 循环 ReAct，用 LangGraph StateGraph
> 重写为显式状态机，补齐「LangGraph 框架」这一简历硬缺口。
> 性质：**增量模块**，不动 `api/agent.py`、`api/service.py` 与 `libs/` 源码。

---

## 改造前 / 改造后 / 价值

| 维度 | 改造前（api/agent.py） | 改造后（api/langgraph_agent.py） | 价值 |
|---|---|---|---|
| 编排方式 | 手写 `for i in range(max_iter)` + 正则解析 Action | LangGraph StateGraph 显式状态机（agent/tools 两节点 + 条件边） | 可讲清状态机/条件边/节点编排 |
| 工具执行 | `kb_search` 内检索+生成耦合（每次调工具都生成一次答案） | tools 节点纯检索，生成收敛到 agent 节点 | 消除双重 LLM 调用，语义更清晰 |
| 决策格式 | 自由文本 ReAct（正则脆弱，易脑补 Observation） | JSON 约束输出（`{tool,input}` / `{answer}`） | 条件边判断确定性，不再靠正则 |
| 状态管理 | 局部变量 `scratch`/`observations` 手动拼接 | `messages` + `add_messages` 归约器 + `iterations` | 状态可追踪、可回放、可 checkpoint |
| 框架 | 无（纯手写） | LangGraph 0.2.76（兼容 langchain-core 0.2.43） | 命中 JD「熟悉 LangGraph」硬要求 |
| 循环保护 | `max_iter` 手动 break | 条件边 + `iterations < 5` + `recursion_limit` | 双保险防死循环 |

---

## 关键工程决策

### 1. 为什么手写 StateGraph，而非 `create_react_agent`

项目 LLM 是 kotaemon 封装的 `ChatOpenAI`，**没有 `bind_tools`**（原生 tool
calling 不可用），而 `create_react_agent` 依赖 `bind_tools` 构造，走不通。
手写状态机反而更有价值：显式状态 + 条件边 + 节点编排，是面试时能讲清
LangGraph 核心机制的最佳素材。

### 2. 两套消息体系的桥接（踩坑记录）

LangGraph 图状态用 `langchain_core` 消息（`SystemMessage`/`AIMessage`/
`ToolMessage`），而 kotaemon 的 `ChatOpenAI.invoke` 只认 `kotaemon.base`
消息（其 `prepare_message` 内部调 `to_openai_format`）。

**直接传 langchain 消息会抛 `AttributeError: 'SystemMessage' object has no
attribute 'to_openai_format'`**。解决方案：`_to_kotaemon_message` 在调 LLM
前做显式类型转换（SystemMessage→KSystemMessage、ToolMessage→KHumanMessage
带 `[检索结果]` 前缀、AIMessage→KAIMessage）。

### 3. 工具节点只检索、不生成

原 `kb_search` 是「检索 + LLM 生成 + 拼引用」三位一体，若直接当 LangGraph
工具用，每轮 tools 节点会触发一次完整 RAG 生成（双重 LLM 调用 + 极慢）。
重构后 tools 节点复用 `service.retriever`（混合检索 + bge 重排）做纯检索，
生成统一由 agent 节点完成，语义更贴近 LangGraph 标准范式。

---

## 图结构

```
START
  │
  ▼
agent ──条件边 _should_continue──┬─ 需检索 & iterations<5 ──► tools ──┐
  ▲                               │                                   │
  └───────────────────────────────┘ ◄─────────────────────────────────┘
                信息足够 / 超迭代 ──► END
```

- 节点：`agent`（LLM 决策）、`tools`（纯检索）
- 条件边：`_should_continue` 根据最后一条 AIMessage 是否含 `tool` 字段，
  且未超 `MAX_TOOL_ITERATIONS=5`，决定回 `tools` 还是 `END`。

---

## 验证结果（真实运行）

| 测试项 | 结果 |
|---|---|
| 图构建 | `['__start__', 'agent', 'tools', '__end__']` ✅ |
| 工具清单 | 4 类（规章/设备/检修/应急）✅ |
| 条件边逻辑 | tools / end / 超迭代 三态正确 ✅ |
| 端到端问答 | 正确回答信号系统五级维修及周期，带条款号出处（交运规〔2019〕8号 第九/十三条），耗时 ~36s（含首次 reranker 加载）✅ |

---

## 使用方式

```powershell
# 直接自测
.\.venv\Scripts\python.exe -m api.langgraph_agent

# 冒烟测试（图构建 + 条件边 + 端到端）
.\.venv\Scripts\python.exe scripts\test_langgraph.py

# API 切换：USE_LANGGRAPH=1 走 LangGraph 版，默认走原 ReAct 版
$env:USE_LANGGRAPH="1"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
```

---

## 简历可用措辞（供包装参考）

> 用 LangGraph StateGraph 将手写 ReAct 循环重构为显式状态机（agent/tools
> 双节点 + 条件边 + add_messages 状态归约器），通过 JSON 约束 LLM 决策输出、
> 将工具执行收敛为纯检索，消除双重 LLM 调用；设计两套消息体系桥接适配层，
> 解决 LangGraph 与 kotaemon LLM 的消息类型不兼容问题。真实问答验证条款号
> 定位与出处可溯。
