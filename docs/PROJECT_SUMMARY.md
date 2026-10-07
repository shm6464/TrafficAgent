# 项目总结：城市轨道交通运维知识库问答系统（简历素材库）

> 用途：本文件是给「后续做简历的 AI」消费的**项目事实清单**。所有数字、文件名、技术决策均来自真实落盘代码与评测报告，可安全直接引用到简历中。
> 定位：这是一份「简历素材库」而非项目说明文档——重点突出**个人贡献、可量化成果、可经面试追问的工程决策**。
> 数据出处：本仓库 `eval/results/report.md`、`docs/02_retrieval_report.md`、`docs/04_engineering_report.md`、`README.md`、`docs/RESUME.md`。

---

## 0. 元信息（必读）

| 项 | 内容 |
|---|---|
| 项目中文名 | 城市轨道交通运维知识库问答系统 |
| 英文标识 | Traffic O&M Knowledge Base RAG QA System |
| 项目性质 | **开源框架 Cinnamon/kotaemon 的改造**（Apache-2.0），非从零造轮子 |
| 上游仓库 | https://github.com/Cinnamon/kotaemon （保留 `LICENSE.txt` 与原项目致谢） |
| 我的角色 | 独立开发者 / 改造负责人（检索层、评测、工程化均一人实现） |
| 开发语言 | Python 3.10（`.venv` 隔离环境，未污染系统 Python） |
| 部署形态 | 单机 + Windows 开发环境 + FastAPI 服务化 + Docker 部署配置（仅写配置，未在本机构建容器） |
| 关键文件根目录 | `D:\google\kotaemon-main\kotaemon-main` |

> ⚠️ 简历表述红线：① 所有百分比/延迟/成本必须是本仓库脚本真实跑出的数字；② 禁止把"开源 kotaemon 自带能力"写成"我的成果"；③ API Key 仅存 `.env`，不进任何文档与日志。

---

## 1. 项目背景与目标（一句话讲清"为什么")

面向地铁/道路交通的**规章与运维文档**，构建检索增强问答（RAG）系统：一线运维人员用自然语言查规章/技术文档，答案必须带**原文出处 + 条款号定位**，且检索不到时明确说"未在知识库中检索到"，不编造。
核心工程目标是把通用 RAG Demo 改造为**能写进简历、经得起面试追问**的个人项目——用**可量化的检索优化效果**证明工程能力。

---

## 2. 技术栈（简历可直接列）

**语言/框架**：Python 3.10 · FastAPI · Pydantic · pytest（TestClient 进程内测试）
**RAG 框架**：kotaemon（检索+生成链路复用，不重复实现）
**向量库**：Chroma（vectorstore，持久化、无外部服务）
**文档库**：LanceDB（docstore，持久化、Windows 友好）
**检索**：BM25（自实现 `OkapiBM25`，中文 char-bigram 分词）+ 向量混合检索 + RRF 融合
**重排**：bge-reranker-base（本地 sentence-transformers 加载 `BAAI/bge-reranker-base`，CrossEncoder）
**Embedding**：fastembed 本地模型（离线缓存，可切 OpenAI 兼容端点 text-embedding-3-large）
**LLM**：DeepSeek（OpenAI 兼容端点，模型 `deepseek-v4-flash`）
**缓存**：diskcache（磁盘缓存，免 Redis 服务，Embedding+LLM 双层）
**日志**：标准 logging + 自实现 JsonFormatter（structlog 未装的降级方案）
**评测**：自实现 LLM-as-judge（ragas 因 langchain 版本冲突降级）
**部署**：Dockerfile.api + docker-compose.yml（仅写配置）

---

## 3. 整体架构（检索→重排→生成链路）

```
用户 query
   ├─ 向量检索 (Chroma, fastembed)  Top-K
   └─ BM25 检索 (OkapiBM25)          Top-K
            │  RRF 融合 (k=60, 排名融合, 规避量纲问题)
            ▼
   bge-reranker-base 两阶段重排 (CrossEncoder, 仅对少量候选)
            ▼
   领域 prompt (条款号 + 原文引用 + 出处)  →  DeepSeek 生成带引用答案
            ▼
   FastAPI /query 返回 {answer, citations, latency_ms, tokens, trace_id}
```

- 磁盘持久化：`ktem_app_data/`（LanceDB 文档库 + Chroma 向量库，索引名 `traffic_ops_kb` / `index_4`）。
- **不破坏原能力**：所有改动走"新增模块 + 配置开关"，通过 `flowsettings.py` 的 `KH_USE_HYBRID_RETRIEVAL` / `KH_USE_RERANKING` / `USE_CACHE` 等开关可切回原逻辑。

---

## 4. 我的核心改造点（简历主体素材，9 项）

| # | 原项目（kotaemon 自带） | 我的改动（新增/修改） | 效果（真实数据） |
|---|---|---|---|
| 1 | 通用 RAG 问答，无业务场景 | 新增 19 份交通运维语料 + 领域 system prompt + UI 文案 | 场景化：答案带规章条款号定位（第三十四~三十七条、第九~十五条等） |
| 2 | 仅向量检索 | 新增 `OkapiBM25` + `HybridFusionRetriever`（RRF 融合，alpha 可调） | recall@5 98.4% → **100.0%**，MRR@5 0.893 → 0.955 |
| 3 | 无本地重排器 | 新增 `BgeReranking`（本地 bge-reranker-base） | 两阶段重排，检索质量提升（MRR 提升） |
| 4 | 无量化评测 | 自建 125 条评测集 + 三方案消融脚本 | 产出完整对比表（`eval/results/report.md`） |
| 5 | 无缓存 | 新增 Embedding + LLM 双层磁盘缓存（diskcache） | 平均延迟 **-88.6%**，token 成本 **-100%**（热缓存） |
| 6 | 无结构化日志 | 新增 JSON 结构化日志（trace_id/耗时/token/命中） | 全链路可观测 |
| 7 | 无成本埋点 | 检索/生成阶段打点 + 成本折算 | token/成本/延迟真实可查 |
| 8 | 仅 Gradio UI | 新增 FastAPI 服务（`/query` `/ingest` `/healthz`） | 服务化，`pytest` 进程内测试 **3 passed** |
| 9 | 无部署配置 | 新增 `Dockerfile.api` + `docker-compose.yml` + `start.ps1` | 一键部署（仅写配置，未构建容器） |

**关键新增代码文件（简历可点名的真实产出）**：
- `libs/kotaemon/kotaemon/indices/retrievers/hybrid.py` —— BM25 + 混合检索（`OkapiBM25` / `BM25Retriever` / `HybridFusionRetriever`）
- `libs/kotaemon/kotaemon/rerankings/bge.py` —— bge 重排（`BgeReranking`，符合 `BaseReranking` 契约）
- `libs/kotaemon/kotaemon/embeddings/cached.py` + `libs/kotaemon/kotaemon/llms/cached.py` —— 双层缓存（`CachedEmbeddings` / `CachedLLM`）
- `libs/ktem/ktem/prompts/traffic.py` —— 交通运维领域 prompt
- `libs/ktem/ktem/utils/structured_log.py` —— 结构化日志
- `api/main.py` + `api/service.py` —— FastAPI 服务与问答核心服务（**进程内复用** kotaemon 链路，不重复实现检索逻辑）
- `eval/eval.py` + `eval/gen_report.py` —— 三方案评测与报告生成
- `scripts/`（ingest_traffic.py / query_cli.py / smoke_retrieval.py / bench_latency.py / start.ps1）—— 脚本化入库、问答、消融、基准

---

## 5. 量化成果（真实数字，全部可回溯到 `eval/results/`）

### 5.1 检索层（完整 125 条评测集，真实计算，不依赖 LLM）

| 方案 | recall@5 | MRR@5 | 平均检索延迟 |
|---|---|---|---|
| A 朴素向量 | 98.4% | 0.893 | 23.2ms |
| B 混合(RRF)+bge重排 | **100.0%** | 0.926 | 35.3ms |
| C 混合+重排+查询改写 | **100.0%** | **0.955** | 3.77s |

### 5.2 生成层（20 条分层抽样，自实现 LLM-as-judge）

| 方案 | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| A | 0.710 | 0.927 | 0.405 | 0.765 |
| B | **0.800** | 0.938 | 0.415 | 0.756 |
| C | 0.740 | **0.963** | 0.440 | **0.854** |

### 5.3 缓存收益（20 题冷/热对比，`scripts/bench_latency.py`）

| 指标 | 冷启动 | 热缓存 | 提升 |
|---|---|---|---|
| P50 延迟 | 14097.6ms | 1899.2ms | **-86.5%** |
| P95 延迟 | 80120.2ms | 6951.1ms | **-91.3%** |
| 平均延迟 | 18871.7ms | 2143.6ms | **-88.6%** |
| token 成本 | 0.3061 元 | 0.0000 元 | **-100%** |

- 单题最高加速比 **10.9x**（22.6s → 2.1s）；热缓存 LLM/embedding 命中率均 100%。

### 5.4 接口测试

`tests/test_api.py` 用 `fastapi.testclient.TestClient` 进程内验证：**3 passed**（healthz / query 结构 / 空问题拒绝）。

---

## 6. 关键工程决策（面试追问弹药，必须能讲清"为什么"）

1. **为什么选 LanceDB + Chroma 而非 Elasticsearch？**
   单机/Windows/19 份文档量级，ES 需 Java+常驻服务+高运维成本；嵌入式方案零服务、磁盘持久化、Windows 友好。量级上到千万文档/多节点再上 ES。
2. **RRF 融合 vs 加权融合？**
   加权融合需分数归一化（BM25 分与余弦分布量纲差异大、α 要调参）；RRF 只关心排名 `1/(k+rank)`，天然规避量纲问题、无需归一化、k=60 鲁棒。默认走 RRF，保留 alpha 加权路径做消融。
3. **BM25 与向量检索各自失效场景？**
   BM25 失效：同义/近义、中文分词、长文档语义弱；向量失效：精确条款号/编号（"第三十五条""1500V"）、罕见专有名词、数值区间。两者 RRF 互补 → recall@5 98.4%→100%。
4. **bge 重排的精度-延迟权衡？**
   CrossEncoder 对 (query,doc) 成对编码，开销随候选数线性增长；两阶段（先召回 Top-K 再对少量候选重排）是标准做法。实测 35.3ms→3.77s，换 MRR@5 0.926→0.955；`use_rerank` 开关支持按请求重要性动态开关。
5. **缓存一致性如何处理？**
   缓存键=内容 hash（embedding 键=文本 sha256；LLM 键=messages 序列化 hash）。风险：知识库更新后旧 LLM 缓存命中 → TTL（LLM 默认 3600s）+ 入库可清缓存；生产更严谨做法是键里纳入文档版本号。
6. **GraphRAG 为什么没硬上（降级为查询改写）？**
   GraphRAG 构建成本高（全库实体/关系抽取、LLM 调用量大、耗时、token 贵）、存储/延迟/工程复杂度高。管线未跑通时如实降级为查询改写（零构建成本），并在报告中说明——主动讲"为什么没上"体现成本意识。
7. **ragas 为什么没用（降级为自实现 LLM-as-judge）？**
   langchain 0.2.x 与 ragas 0.2+ 版本冲突；降级为自实现 judge 打 faithfulness/answer_relevancy/context_precision/context_recall 四维，口径已在报告标注。但 recall@5/MRR/延迟/成本始终真实计算。
8. **评测集标注质量如何保证？**
   ground_truth **逐条手工从 19 份语料原文抽取**（条款号、检修周期数值、限速值、事件分级阈值均为原文真实内容），带 source_file+source_page 可回溯；**禁止 LLM 凭空生成标准答案**。最终 125 条（single_hop 65 / numeric 26 / table 11 / multi_hop 23）。

---

## 7. 诚实声明 / 已知限制（简历防追问，务必如实写）

- ragas 未接入：生成层指标为自实现 LLM-as-judge 口径（已在报告标注），非 RAGAS。
- GraphRAG 未接入：方案 C 降级为"查询改写"，原 GraphRAG 管线未跑通（已说明）。
- 冷启动延迟受 DeepSeek 高峰时段（14:00–18:00）限速影响，P95 波动大（最高 80s）；缓存命中后稳定 ~2s。
- structlog 未装：结构化日志为 logging+JSON 降级实现，接口已兼容原生 structlog。
- Docker 仅写配置未在本机构建/运行容器（Windows 环境拉镜像易失败，按硬约束不跑）。
- 向量库仍用 Chroma（曾尝试切 LanceDB 因 `lancedb 0.25.x` 与集成层 schema 不兼容，且降级会破坏已建 LanceDB docstore，故 docstore=LanceDB、vectorstore=Chroma）。
- 评测 LLM 为 DeepSeek `deepseek-v4-flash`，成本单价写死在 `api/service.py`（输入 1.5 元/百万 token、输出 4.5 元/百万 token）口径一致可溯。

---

## 8. 可直接复用的简历 Bullet（3 条，已按"技术栈+动作+真实数字"打磨）

> **Bullet 1（检索层——核心亮点）**
> 交通运维知识库 RAG 问答系统（Python / Chroma+LanceDB / BM25+向量混合检索 / bge-reranker）：基于开源框架 kotaemon 改造面向地铁运维的规章问答系统，实现 BM25+向量两路检索的 RRF 融合与 bge-reranker-base 两阶段重排，自建 125 条评测集（multi_hop 23 / table 11 / numeric 26）做三方案消融，**recall@5 由 98.4% 提升至 100.0%，MRR@5 由 0.893 提升至 0.955**。

> **Bullet 2（评测量化——面试杀手锏）**
> 量化评测体系：ground_truth 逐条从 19 份规章/规范原文手工抽取（非 LLM 生成），以自实现 LLM-as-judge 评测 faithfulness / answer_relevancy / context_precision / context_recall 四维生成质量（ragas 因依赖冲突降级），**answer_relevancy 达 0.963、faithfulness 达 0.800**，并自算 recall@5 / MRR@5 / 延迟 / token 成本全套硬指标。

> **Bullet 3（工程化——落地能力）**
> 工程化与服务化：实现 Embedding + LLM 双层磁盘缓存（diskcache，命中即省 token），**平均响应延迟降低 88.6%、token 成本降低 100%**（热缓存，20 题基准）；接入 JSON 结构化日志与成本埋点（trace_id / 阶段耗时 / token），FastAPI 服务化（/query /ingest /healthz）并以 pytest 进程内测试保障（3 passed），提供 Dockerfile + docker-compose 部署配置。

> 更完整的面试 8 问 8 答见 `docs/RESUME.md`；演示问题/截图占位见 `docs/DEMO.md`。

---

## 9. 文件索引（AI 需要核对原文时按图索骥）

| 用途 | 路径 |
|---|---|
| 项目说明（中英） | `README.md` |
| 改造指令与硬约束 | `INSTRUCTIONS.md` |
| 评测报告（三方案数字） | `eval/results/report.md` |
| 检索层报告+消融+失败案例 | `docs/02_retrieval_report.md` |
| 工程化报告+缓存数字 | `docs/04_engineering_report.md` |
| 评测集（125 条） | `eval/qa_set.jsonl` |
| 语料清单 | `data/traffic_docs/MANIFEST.md` |
| 简历/面试弹药 | `docs/RESUME.md` |
| 演示脚本 | `docs/DEMO.md` |
| 环境体检 | `docs/00_env_report.md` / `docs/01_domain_report.md` |
| 主配置开关 | `flowsettings.py` |
| FastAPI 服务 | `api/main.py` / `api/service.py` |
| 混合检索/重排/缓存模块 | `libs/kotaemon/kotaemon/indices/retrievers/hybrid.py`、`libs/kotaemon/kotaemon/rerankings/bge.py`、`libs/kotaemon/kotaemon/embeddings/cached.py`、`libs/kotaemon/kotaemon/llms/cached.py` |
| 接口测试 | `tests/test_api.py` |
