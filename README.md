<div align="center">

# TrafficAgent — 城市轨道交通运维知识库问答系统

**Traffic O\&M Knowledge Base RAG QA System**

TrafficAgent 是一个面向城市轨道交通运维场景的检索增强问答（RAG）系统：一线运维人员用自然语言查规章 / 技术文档，答案带**原文出处 + 条款号定位**，并以**可量化的检索优化效果**与**完整 Agent / LangGraph / MCP 能力栈**支撑生产级问答。

</div>

***

## 项目定位

面向地铁 / 道路交通的**规章与运维文档**构建检索增强问答（RAG）系统，核心能力：

- **两阶段检索**：BM25 + 向量混合检索（RRF 融合）+ bge-reranker 交叉编码重排
- **领域化**：交通运维专用 system prompt，答案强制带条款号 + 原文引用 + 文档出处，检索不到明确说「未在知识库中检索到」，不编造
- **可量化**：自建 125 条评测集，三方案消融对比（recall / MRR / 延迟 / 成本 / 生成质量）
- **Agent 化**：ReAct Agent 多步检索（规章/设备/检修/应急四类工具）+ 多轮记忆（内存/Redis 可切换）
- **LangGraph**：用 StateGraph 显式状态机重构 ReAct 循环，消除双重 LLM 调用
- **MCP 化**：检索/问答/Agent 封装为 MCP 工具，可被 Claude Code / Cursor 等外部 Agent 调用
- **工程化**：Embedding + LLM 双层磁盘缓存、结构化日志、成本埋点、FastAPI 服务化、Docker 部署配置

**技术栈**
`Python 3.10` · `Chroma`(向量) + `LanceDB`(文档) · `FastEmbed(bge-small-zh-v1.5)`(向量化) / `bge-reranker-base`(重排) · `BM25+向量混合检索(RRF)` · `FastAPI` · `LangGraph` · `MCP(FastMCP)` · `diskcache` · `DeepSeek`(经 OpenAI 兼容通道)

***

## 界面展示

TrafficAgent 提供对话式运维问答界面，支持多轮对话、ReAct 推理过程可视化与原文出处引用。

**主界面**：会话管理、欢迎语与运维场景建议问题。

![主界面](docs/screenshots/01_chat_home.png)

**自我介绍与能力说明**：Agent 主动说明可覆盖的运维领域（应急响应、设备维护、信号系统、系统概览等）。

![自我介绍](docs/screenshots/02_self_intro.png)

**信号系统问答**：覆盖 CBTC 原理、ATP/ATO/ATS 各层级职责。

![信号系统](docs/screenshots/03_signal_system.png)

**多步检索（ReAct）**：用户询问不同级别应急响应措施时，Agent 执行 `search_emergency` 工具检索知识库，Thought → Action → Observation 推理链完整可见。

![应急响应 ReAct 检索](docs/screenshots/04_emergency_react.png)

**分级响应详解（IV 级 / III 级）**：

![IV级与III级响应](docs/screenshots/05_emergency_levels_1.png)

**分级响应详解（II 级 / I 级）与总结**：

![II级与I级响应及总结](docs/screenshots/06_emergency_levels_2.png)

**完整回答**：结构化呈现四个响应级别并给出总结，答案有据可查。

![应急响应总结](docs/screenshots/07_emergency_summary.png)

***

## 架构

```
                         ┌─────────────────────────────────────┐
                         │         FastAPI (api/main.py)         │
                         │ /query  /ingest  /healthz  /agent ...  │
                         └──────────────────┬──────────────────┘
                                            │
       ┌────────────────────────────────────┼──────────────────────────┐
       │                                    │                          │
       ▼                                    ▼                          ▼
┌───────────────┐              ┌──────────────────┐       ┌──────────────────┐
│  ReAct Agent  │              │  LangGraph Agent │       │   MCP 工具服务    │
│ (api/agent.py)│              │(langgraph_agent) │       │ (api/mcp_server) │
│ 手写 while 循环│              │ StateGraph 状态机 │       │ kb_search/kb_qa/ │
│ 多工具+多轮记忆│              │ agent/tools 节点  │       │ kb_agent 三工具   │
└───────┬───────┘              └────────┬─────────┘       └────────┬─────────┘
        │                                │                          │
        └──────────────┬─────────────────┘                          │
                       ▼                                             │
             ┌──────────────────┐◄──────────────────────────────────┘
             │  api/service.py   │
             │ 检索+生成核心链路   │
             │ CachedEmbeddings → │
             │ HybridFusionRetriever → BgeReranking │
             │ → CachedLLM        │
             └────────┬───────────┘
                      │
       ┌──────────────┼──────────────┐
       │  Chroma 向量库 │ LanceDB 文档库 │  ← ktem_app_data/（磁盘持久化）
       └──────────────┴──────────────┘
```

**检索链路**：`向量 Top-K` + `BM25 Top-K` → **RRF 融合** → **bge-reranker 重排** → 送入 LLM 生成带引用答案。
**Agent 链路**：LLM 决策（调哪类工具）→ 工具检索（按文档类别过滤）→ 观察回填 → 迭代 → 最终答案。

***

## 功能实现（实现方式 / 价值）

### A. 领域化与语料

| #  | 实现方式                                                                                               | 价值                         |
| -- | -------------------------------------------------------------------------------------------------- | -------------------------- |
| A1 | 新增 `data/traffic_docs/` 19 份交通运维语料（规章 8 / 技术文档 8 / 规范 2 / 手册 1），含 `MANIFEST.md` 溯源说明               | 可验证领域知识库，支持条款号定位、表格解析、多跳问答 |
| A2 | 新增 `ktem/prompts/traffic.py`（系统提示+QA 模板：强制条款号+原文引用+出处，检索不到明说「未在知识库中检索到」）；UI 占位改中文；首页建议问题改为 5 个运维场景 | 答案可审计、可溯源，杜绝编造             |

### B. 检索层

| #  | 实现方式                                                                                                                                 | 价值               |
| -- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| B1 | 新增 `indices/retrievers/hybrid.py`：纯 Python `OkapiBM25`（中文 char-bigram 分词）+ `BM25Retriever` + `HybridFusionRetriever`（RRF 融合，亦支持加权融合） | 中文无需依赖英文分词器；召回更稳 |
| B2 | 实现 `LanceDBDocStore.get_all()` 供 BM25 建索引                                                                                            | 打通混合检索语料来源       |
| B3 | `ktem/index/file/pipelines.py` 注入 `hybrid_retrieval` 节点，由 `KH_USE_HYBRID_RETRIEVAL` 开关切换（关则回退向量）                                     | 可灰度、可 A/B        |
| B4 | 真实评测：recall\@5 **98.4% → 100.0%**；MRR\@5 **0.893 → 0.926**（混合+重排，方案 B），**0.955**（混合+重排+查询改写，方案 C）                                    | 量化证明混合检索价值       |

### C. 重排与缓存

| #  | 实现方式                                                                                                         | 价值                                    |
| -- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| C1 | 新增 `rerankings/bge.py`（`BgeReranking`，本地 `bge-reranker-base`，加载失败降级保序）；注册进 `ktem/rerankings/manager.py`      | 离线重排、零 API 成本、隐私可控                    |
| C2 | 新增 `embeddings/cached.py`（`CachedEmbeddings`，diskcache，按文本 sha256 缓存）、`llms/cached.py`（`CachedLLM`，LLM 响应缓存） | 热缓存平均延迟 **-88.6%**、token 成本 **-100%** |
| C3 | `vectorindex.py` 改为直接 `.run()`（绕开 Windows 上共享缓存/上下文管理卡死）                                                     | 入库在 Windows 稳定可跑                      |
| C4 | `lancedb` 异常处理扩展为 `(FileNotFoundError, ValueError)`                                                          | 消除版本相关「表不存在」报错                        |

### D. 生成与可观测性

| #  | 实现方式                                                                                                                                | 价值           |
| -- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| D1 | `api/service.py` 全链路埋点：trace\_id / 阶段耗时 / 检索命中数 / token / 缓存命中；`ktem/utils/structured_log.py` 输出 JSON 日志（structlog 缺失时降级标准 logging） | 质量与成本可量化、可排查 |
| D2 | `reasoning/simple.py` citation 默认 highlight 关闭（`"off"`），降低低配 LLM 请求；领域 prompt 开关覆盖默认                                                | 稳定 + 领域化     |

### E. Agent / LangGraph / MCP

| #  | 实现方式                                                                                                                              | 价值                      |
| -- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| E1 | `api/agent.py` 手写 ReAct：`TrafficAgent`，四类工具（规章/设备/检修/应急，按文件名分类），token 级流式推理+答案；专门处理 DeepSeek「脑补完整循环」——优先取 Action、截断伪造 Observation | 多步检索 + 多轮记忆，真实可用        |
| E2 | `api/langgraph_agent.py`：用 `StateGraph` 重写 ReAct 为显式状态机（agent↔tools 条件边），**tools 节点只检索不生成**，消除双重 LLM 调用；JSON 约束决策 + 健壮解析          | 补齐 LangGraph 框架能力，架构可讲清 |
| E3 | `api/memory_store.py`：记忆抽象层，默认内存、`MEMORY_BACKEND=redis` 切换 Redis（TTL 自动过期），Redis 不可用优雅降级内存                                        | 多轮会话可水平扩展               |
| E4 | `api/mcp_server.py`：FastMCP 暴露 `kb_search`/`kb_qa`/`kb_agent` 三工具（stdio），配 `mcp_config.example.json`                              | 知识库作为工具被外部 Agent 调用     |

### F. 工程化与部署

| #  | 实现方式                                                                                                                  | 价值           |
| -- | --------------------------------------------------------------------------------------------------------------------- | ------------ |
| F1 | `api/main.py` 完整 FastAPI：`/query` `/ingest` `/agent` `/agent/stream` `/healthz`；本地 Swagger UI（不依赖 CDN，国内可用）；根路径运维风交互页 | 服务化、可测试      |
| F2 | `Dockerfile.api` + `docker-compose.yml`（API 镜像，持久化卷，缓存/检索开关经 env 注入）                                                  | 一键部署配置齐备     |
| F3 | `manager.py` 模型入库改为 upsert：让 `.env` 成为每次启动事实源，覆盖库内旧 spec                                                              | 换 Key/模型即时生效 |

### G. 评测体系

| #  | 实现方式                                                                                                            | 价值                            |
| -- | --------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| G1 | `eval/qa_set.jsonl` **125 条**（single\_hop 65 / numeric 26 / table 11 / multi\_hop 23），ground\_truth 逐条从语料原文手工抽取 | 评测可信、可回溯，不靠 LLM 编答案           |
| G2 | `eval/eval.py` + `gen_report.py` 三方案消融；`scripts/bench_latency.py` 冷/热缓存对比                                       | 产出完整 `eval/results/report.md` |
| G3 | `scripts/` + `tests/test_api.py`：ingest/query CLI、smoke 基线、Agent/LangGraph/记忆/流式/工具测试、进程内 API 测试                | 工程化验收闭环                       |

***

## 评测结果

> 数字均来自 `eval/eval.py` 真实运行输出与 `scripts/bench_latency.py` 基准，未手工估算。详见 `eval/results/report.md`。

### 检索层（完整 125 条评测集）

| 方案               | recall\@5  | MRR\@5    | 平均检索延迟 |
| ---------------- | ---------- | --------- | ------ |
| A 朴素向量（基线）       | 98.4%      | 0.893     | 23.2ms |
| B 混合(RRF)+bge 重排 | **100.0%** | 0.926     | 35.3ms |
| C 混合+重排+查询改写     | **100.0%** | **0.955** | 3.77s  |

### 生成层（20 条分层抽样，自实现 LLM-as-judge）

| 方案 | faithfulness | answer\_relevancy | context\_recall |
| -- | ------------ | ----------------- | --------------- |
| A  | 0.710        | 0.927             | 0.765           |
| B  | **0.800**    | 0.938             | 0.756           |
| C  | 0.740        | **0.963**         | **0.854**       |

### 缓存收益（20 题冷/热对比，`scripts/bench_latency.py`）

| 指标       | 冷启动       | 热缓存      | 提升         |
| -------- | --------- | -------- | ---------- |
| P50 延迟   | 14097.6ms | 1899.2ms | **-86.5%** |
| P95 延迟   | 80120.2ms | 6951.1ms | **-91.3%** |
| 平均延迟     | 18871.7ms | 2143.6ms | **-88.6%** |
| token 成本 | 0.3061 元  | 0.0000 元 | **-100%**  |

- 评测集 125 条（single\_hop 65 / numeric 26 / table 11 / multi\_hop 23），ground\_truth 逐条从语料原文手工抽取，未用 LLM 凭空生成。
- 生成层指标采用自实现 LLM-as-judge（ragas 因 langchain 0.2.x 版本冲突降级，口径已在报告标注）。

***

## 快速开始（PowerShell）

```powershell
# 0. 环境（Python 3.10，已有 .venv）
cd TrafficAgent
.\.venv\Scripts\activate

# 0.1 安装依赖（主体依赖走 pyproject，改造新增依赖走 requirements.txt）
pip install -e .
pip install -r requirements.txt

# 1. 配置模型端点（.env，Key 不入库）
#    复制 .env.example 为 .env，填入 OPENAI_API_BASE / OPENAI_API_KEY / OPENAI_CHAT_MODEL

# 2. 入库（脚本化，不起 UI）
python scripts\ingest_traffic.py

# 3. 命令行问答
python scripts\query_cli.py

# 4. 启动 API 服务
.\scripts\start.ps1            # 默认 8000 端口，不起 Docker

# 5. ReAct Agent 问答（多步检索 + 多轮记忆）
python -m api.agent

# 6. LangGraph 版 Agent（显式状态机）
python -m api.langgraph_agent

# 7. MCP 工具服务（供外部 Agent 调用）
python api\mcp_server.py       # stdio 模式

# 8. 跑评测（检索指标秒级，生成指标依赖 LLM 较慢）
python eval\eval.py --scheme all

# 9. 延迟基准（缓存冷/热对比）
python scripts\bench_latency.py --n 20

# 10. 接口测试（进程内 TestClient，不起服务）
python -m pytest tests\test_api.py -v
```

***

## MCP 接入（可选）

把知识库能力作为 MCP 工具开放给外部 Agent：

```powershell
# 1. 直接运行（stdio 模式，供 MCP 客户端拉起）
.\.venv\Scripts\python.exe api\mcp_server.py

# 2. 在支持 MCP 的客户端（如 Claude Code / Cursor）配置 mcp_config.example.json
#    暴露三个工具：
#      kb_search  知识库纯检索（原文片段 + 出处）
#      kb_qa      知识库问答（混合检索 + 重排 + 生成）
#      kb_agent   ReAct Agent 问答（多步检索 + 多轮记忆）
```

***

## 目录结构

```
TrafficAgent/
├── api/                      # FastAPI 服务 + Agent + LangGraph + MCP
│   ├── main.py               # 服务入口（/query /ingest /agent /agent/stream /healthz）
│   ├── service.py            # 检索+生成核心服务（复用链路 + 缓存 + 埋点 + 重试）
│   ├── agent.py              # ReAct Agent（手写循环 + 四类工具 + 多轮记忆）
│   ├── langgraph_agent.py    # LangGraph 状态机版 Agent（StateGraph）
│   ├── mcp_server.py         # MCP 工具服务（kb_search/kb_qa/kb_agent）
│   ├── memory_store.py       # 多轮记忆（内存/Redis 可切换）
│   └── static/               # 本地 Swagger UI + 交互界面
├── data/traffic_docs/        # 19 份交通运维语料 + MANIFEST.md
├── eval/                     # 评测集 + 评测脚本 + 结果
│   ├── qa_set.jsonl          # 125 条评测集
│   ├── eval.py               # 三方案评测脚本
│   └── results/              # metrics CSV/JSON + report.md
├── libs/kotaemon/kotaemon/   # 框架层（hybrid.py / bge.py / cached.py）
├── libs/ktem/ktem/           # 应用层（领域 prompt / 结构化日志）
├── scripts/                  # ingest / query_cli / smoke / bench / test_* / start.ps1
├── tests/test_api.py         # 接口测试
├── docs/                     # 各 Phase 报告（00~06）+ RESUME/DEMO
│   └── screenshots/          # 界面展示截图
├── flowsettings.py           # 配置开关（缓存/混合检索/重排/领域prompt）
├── requirements.txt          # 改造新增依赖（torch / sentence-transformers / diskcache / langgraph / mcp）
├── mcp_config.example.json   # MCP 客户端接入配置示例
├── Dockerfile.api            # 部署配置
└── docker-compose.yml
```

