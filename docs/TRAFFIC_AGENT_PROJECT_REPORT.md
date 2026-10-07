# 城市轨道交通运维知识库问答系统 · 项目报告

> 改造基线：[Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon)（Apache-2.0）
> 子系统重点：**TrafficAgent 交互增强**（ReAct Agent + token 级流式 + 历史会话 + 可插拔记忆）
> 报告生成日期：2026-10-06

---

## 1. 项目概览

面向地铁 / 道路交通的**规章与运维文档**构建检索增强问答（RAG）系统，一线运维人员用自然语言查规章 / 技术文档，答案带**原文出处 + 条款号定位**。

在 kotaemon 基线之上，本项目增量实现了四层能力：

- **RAG 检索**：BM25 + 向量混合检索（RRF 融合）+ bge-reranker 重排
- **ReAct Agent**：多步检索（规章 / 设备 / 检修 / 应急四类工具）+ 多轮记忆（内存 / Redis 可切换）
- **LangGraph 状态机**：用 `StateGraph` 重写 ReAct 循环，消除双重 LLM 调用
- **MCP 工具服务**：检索 / 问答 / Agent 封装为 MCP 工具，可被外部 Agent 调用
- **TrafficAgent 交互系统**（本次报告重点）：基于 FastAPI + SSE 的流式对话界面

技术栈：`Python 3.10` / `Chroma` + `LanceDB` / `BM25` + 向量混合 / `bge-reranker-base` / `FastAPI` / `LangGraph` / `MCP` / `diskcache`；LLM 走**阿里云百炼** OpenAI 兼容端点（`.env` 主导配置）。

---

## 2. 系统架构

```
                         ┌─────────────────────────────────────┐
                         │        FastAPI (api/main.py)          │
                         │  POST /query  POST /ingest GET /healthz│
                         │  POST /agent  POST /agent/stream        │
                         └─────────────────┬─────────────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────┐
        ▼                                  ▼                          ▼
┌──────────────────┐            ┌──────────────────┐       ┌──────────────────┐
│  TrafficAgent     │            │  LangGraph Agent │       │   MCP 工具服务    │
│  (api/agent.py)   │            │ (langgraph_agent)│       │  (api/mcp_server) │
│  ReAct + 四类工具 │            │ StateGraph 状态机 │       │ kb_search/qa/agent│
│  + 多轮记忆        │            │  agent/tools 节点 │       │                    │
└────────┬─────────┘            └────────┬─────────┘       └────────┬─────────┘
         └──────────────────┬─────────────┘                       │
                            ▼                                     │
              ┌──────────────────────┐                           │
              │  api/service.py       │◄──────────────────────────┘
              │ 检索+生成链路（复用）   │
              │ CachedEmbeddings →     │
              │ HybridFusionRetriever  │
              │ → CachedLLM            │
              └─────────┬──────────────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   Chroma 向量库   LanceDB 文档库   ktem_app_data/（磁盘持久化）
```

检索链路：`向量 Top-K` + `BM25 Top-K` → **RRF 融合** → **bge-reranker 重排** → LLM 生成带引用答案。
Agent 链路：LLM 决策（调哪类工具）→ 工具检索（按文档类别过滤）→ 观察回填 → 迭代 → 最终答案。

---

## 3. 核心能力改造清单（改造前 / 改造后 / 价值）

| # | 改造前（kotaemon 基线） | 改造后（本项目增量） | 价值（真实数据） |
|---|---|---|---|
| 1 | 通用 RAG 问答，无业务场景 | 19 份交通运维语料 + 领域 prompt | 场景化：规章条款号定位 |
| 2 | 仅向量检索 | `OkapiBM25` + `HybridFusionRetriever`（RRF） | recall@5 98.4% → **100.0%** |
| 3 | 无本地重排器 | `BgeReranking`（本地 bge-reranker-base） | MRR@5 提升 |
| 4 | 无量化评测 | 自建 125 条评测集 + 三方案消融 | 完整对比表（见 §5） |
| 5 | 无缓存 | Embedding + LLM 双层磁盘缓存（diskcache） | 平均延迟 **-88.6%** |
| 6 | 无结构化日志 | JSON 结构化日志（trace_id/耗时/token） | 全链路可观测 |
| 7 | 无成本埋点 | 检索/生成阶段打点 + 成本折算 | token/成本真实可查 |
| 8 | 仅 Gradio UI | FastAPI 服务（`/query` `/ingest` `/agent` …） | 服务化，pytest 通过 |
| 9 | 无部署配置 | `Dockerfile.api` + `docker-compose.yml` + `start.ps1` | 一键部署配置 |
| 10 | 无交通领域多步 ReAct Agent | 手写 ReAct Agent（四类工具 + 多轮记忆） | 多步检索自动路由 |
| 11 | 本人自研的手写 while 循环 ReAct（接入 LangGraph 前的过渡实现，**非 kotaemon 原有代码**） | LangGraph `StateGraph` 重写 | 消除双重 LLM 调用 |
| 12 | 仅 HTTP 接口 | MCP 工具服务（kb_search/kb_qa/kb_agent） | 可被外部 Agent 调用 |
| 13 | 无独立交互 UI（Swagger） | **TrafficAgent 深色运维风对话界面**（见 §4） | 流式推理 + 历史会话 |

> **溯源说明（简历叙事准确性）**：kotaemon 基线仅提供 RAG 框架，**不含任何 ReAct Agent**。第 10 项的「手写 ReAct Agent」与第 11 项的「手写 while 循环 ReAct」均为本人自研实现；第 11 项是接入 LangGraph 之前的过渡版本，第 12 项 LangGraph `StateGraph` 是对该自研 while 循环的重写，均非 kotaemon 原有组件。

---

## 4. TrafficAgent 交互系统（本次重点改造）

### 4.1 改造前 / 改造后 / 价值

| 维度 | 改造前 | 改造后 | 价值 |
|---|---|---|---|
| 交互字体 | body 15px，阅读偏小 | 统一 **16px**，行高 1.7 | 可读性对齐 DeepSeek 网页端 |
| 历史会话 | 仅内存，刷新丢失 | 左侧 268px 历史面板 + `localStorage` 持久化 | 刷新/重开浏览器保留对话 |
| 删除记录 | 无 | 会话项 hover 删除按钮 + 确认 | 可清理无关会话 |
| 推理展示 | 思考阶段静止转圈（10–20s 无输出） | ReAct 推理**逐 token 流式**（`thought_token`） | 消除"卡顿"静止感 |
| 答案输出 | token 级流式 | token 级流式（保留） | 边生成边看 |
| 会话切换 | 切走再切回内容丢失、输入框锁死 | live 气泡重建 + `activeMsgIndex` 修正 | 切换不丢内容、可继续对话 |
| 多轮记忆 | 全局 SESSIONS 字典 | 可插拔 `MemoryStore`（InMemory / Redis 工厂） | 生产可切 Redis |
| 工具路由 | 单检索工具 | 四类工具按 `file_name` 过滤（规章/设备/检修/应急） | 检索更精准 |
| 模型配置 | 首次启动固化 `sql.db` | `.env` 每次启动覆盖 DB（方案 A） | 换模型只改 `.env` |

### 4.2 关键实现

**前端 `api/static/agent.html`（单文件，SSE）**

- **流式管道**：`POST /agent/stream` → `StreamingResponse` → 前端 `fetch` + `ReadableStream` 分帧解析 `event: xxx` + `data: json`。
- **事件类型**：`thought_token`（推理逐字）/ `action` / `observation` / `final_token`（答案逐字）/ `done` / `status`。
- **实时推理面板 `.reasoning-area`**：等宽暗色 + 左侧高亮边 + 闪烁光标，首秒起即展示模型思考。
- **切换会话不丢内容**：
  - 开始生成时立即把 `{role:"agent", content:"", streaming:true}` 写入 `session.messages`；
  - `renderStreamingMessage()` 切回时重建含 reasoning/trace/answer/status 的完整 live 气泡，挂到 `session.liveBox`；
  - `currentBox()` 优先用当前可见的 `liveBox`，保证 SSE 事件落到可见 DOM；
  - 修复 `activeMsgIndex` 误指向用户消息的根因（详见 §6 故障记录）。
- **持久化**：会话存 `localStorage.trafficAgentState`，刷新保留；页面 reload 时清理 stale `streaming` 脏标记。

**后端 `api/agent.py`**

- `_call_llm` 重构为生成器 `_stream_llm`：保留 3 次重试退避，逐 token `yield`。
- ReAct 阶段 1 遍历 `_stream_llm`，每收到 token 即 `yield {"type":"thought_token", ...}`；LLM 调用失败（空输出）时 `break` 兜底进答案生成。
- 工具调用状态文案更明确：`"正在检索知识库并综合分析（{name}）：{inp}"`，降低"是不是卡了"错觉。

**可插拔记忆 `api/memory_store.py`**

- `InMemoryStore(max_turns)` + `RedisMemoryStore` + `get_memory_store()` 工厂，按配置切换。

**四类工具（`TOOLS`）**

| 工具 | 覆盖知识 |
|---|---|
| `search_regulations` | 管理办法 / 技术规范 / 行车组织 |
| `search_equipment` | CBTC / 牵引供电 / 站台门 / AFC 等设备原理与参数 |
| `search_maintenance` | 检修修程 / 维修分级 / 养护周期 |
| `search_emergency` | 突发事件应急预案 / 火灾处置 / 预案体系 |

---

## 5. 评测结果（真实数据，未编造）

> 来源：`eval/eval.py`、`eval/results/report.md`、`eval/results/agent_vs_rag_20261005.json`。
> 口径：ragas 因 langchain 0.2.x 版本冲突降级为自实现 LLM-as-judge，已在原报告标注。

### 5.1 检索层（完整 125 条评测集）

| 方案 | recall@5 | MRR@5 | 平均检索延迟 |
|---|---|---|---|
| A 朴素向量 | 98.4% | 0.893 | 23.2ms |
| B 混合(RRF)+bge重排 | **100.0%** | 0.926 | 35.3ms |
| C 混合+重排+查询改写 | **100.0%** | **0.955** | 3.77s |

### 5.2 生成层（20 条分层抽样，LLM-as-judge）

| 方案 | faithfulness | answer_relevancy | context_recall |
|---|---|---|---|
| A | 0.710 | 0.927 | 0.765 |
| B | **0.800** | 0.938 | 0.756 |
| C | 0.740 | **0.963** | **0.854** |

### 5.3 缓存收益（20 题冷/热对比）

| 指标 | 冷启动 | 热缓存 | 提升 |
|---|---|---|---|
| P50 延迟 | 14097.6ms | 1899.2ms | **-86.5%** |
| P95 延迟 | 80120.2ms | 6951.1ms | **-91.3%** |
| 平均延迟 | 18871.7ms | 2143.6ms | **-88.6%** |
| token 成本 | 0.3061 元 | 0.0000 元 | **-100%** |

### 5.4 Agent vs RAG（preliminary，小样本 n=3）

> ⚠️ **如实标注**：该评测因晚高峰模型限速被中断，仅完成 3 条 multi_hop 样本（目标 8 条），结论**仅供方向性参考，不能用于简历硬指标**。

| 指标 | RAG | ReAct Agent | 差值 |
|---|---|---|---|
| faithfulness（均值） | 0.367 | 0.633 | **+0.267** |
| answer_relevancy（均值） | 0.700 | 0.833 | **+0.133** |
| 单题延迟 | 约 11–47s | 约 32–41s | Agent 略慢（多步检索） |

样本明细（q017 / q027 / q033，均为 hard 难度 multi_hop）：Agent 在"跨文档对比"（q017 通信 vs 信号维修分级、q033 架修 A vs B）上 faithfulness 显著优于 RAG；q027（ATP/ATO/ATS 层次）relevancy 提升明显。

---

## 6. 故障记录（交互系统）

| 现象 | 根因 | 修复 |
|---|---|---|
| 新建会话发送后无回复、输入框未清空 | `newSession()`/`renderSession()` 未清空输入框；异常绕过 `.finally` 致 `sending` 锁死 | 清空输入框 + 前置步骤 `try/catch` + 统一 `resetInput()` |
| 切换会话后内容消失、无法继续对话 | `send()` 中 `activeMsgIndex` 在 `pushMessage(user)` 前捕获，实际指向**用户消息**；`updateThisAgentMessage` 因 `role!=="agent"` 直接 return，答案/`streaming=false` 从未写入 `session.messages` | `activeMsgIndex` 改为 Agent 消息 push 后赋值；加 `role` 校验；reload 清理 stale `streaming` |
| 思考阶段静止 10–20s 像卡死 | ReAct 决策 LLM 整段收集后才发事件 | `_stream_llm` 逐 token 发 `thought_token`，前端 `.reasoning-area` 实时展示 |

---

## 7. 运行与部署

```powershell
cd D:\google\otaemon-main\kotaemon-main     # 注意实际路径
.\.venv\Scripts\activate

# 1. 配置模型（.env，Key 不入库）
#    OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
#    OPENAI_API_KEY=sk-xxx      OPENAI_CHAT_MODEL=qwen-plus

# 2. 入库
python scripts\ingest_traffic.py

# 3. 启动交互服务（默认 8000 端口）
.\scripts\start.ps1
# 浏览器打开 http://127.0.0.1:8000/

# 4. 接口测试
python -m pytest tests\test_api.py -v
```

> 切换模型：仅改 `.env` 即可（`.env` 主导覆盖 `sql.db`，方案 A 已生效）。

---

## 8. 已知限制与后续规划

- **Agent vs RAG 评测未完成**：晚高峰限速致中断，仅 n=3；需错峰重跑满 8 条以产出稳定结论。
- **ragas 未接入**：langchain 0.2.x 与 ragas 0.2+ 冲突，生成层为自实现 LLM-as-judge 口径。
- **GraphRAG 未接入**：方案 C 降级为"查询改写"。
- **冷启动延迟**：受百炼高峰限速影响，P95 波动大；缓存命中后稳定 ~2s。
- **后续**：错峰跑满 Agent vs RAG 评测 → 接入 GraphRAG 多跳增强 → 原生 structlog → MCP 工具接入评测集做回归。

---

## 9. 简历可用量化指标（取自真实评测）

- 检索 recall@5 **100.0%**（混合+重排+改写），MRR@5 **0.955**
- 平均延迟 **-88.6%**（磁盘缓存命中），token 成本 **-100%**（热缓存）
- 生成 faithfulness 最高 **0.800**（方案 B）/ answer_relevancy **0.963**（方案 C）
- ReAct Agent 相对单跳 RAG：faithfulness **+0.267**、relevancy **+0.133**（preliminary, n=3）
- 交互：ReAct 推理 token 级流式（首秒可见），会话切换不丢内容，历史会话 `localStorage` 持久化

> 指标均来自 `eval/results/` 真实运行产物，未手工估算或编造。
