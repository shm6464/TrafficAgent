# 项目问题 → 解决方案清单

> 目标项目：**城市轨道交通运维知识库问答系统（RAG）**
> 配套文件：`proplem.txt`（你列的问题）、`简历_邵海萌_Agent应用开发实习.docx`（当前简历）、本仓库 `docs/`（RESUME.md / 0x_*.md 报告）
> 本文档逐条给出：**根因/风险 → 解决方案 → 落地动作（命令/话术）**，并按优先级排序行动清单。

---

## 0. 关键发现（读代码后必须优先处理的 3 个雷点）

| # | 发现 | 严重度 | 一句话影响 |
|---|---|---|---|
| F1 | `eval/results/metrics_20261002.csv` 里 `faithfulness / answer_relevancy / context_precision / context_recall` **四列全为 0.0**，且 `gen_ms / prompt_tokens / completion_tokens` 也为 0 | 🔴 高危 | 简历写的 `faithfulness 0.800 / answer_relevancy 0.963` 在该 CSV 里**无任何数据支撑**，面试官要脚本输出会当场穿帮 |
| F2 | 简历写「2026.08–2026.10 ｜ 独立开发」，但 BM25/混合检索/bge重排/双层缓存都是**直接改在 `libs/kotaemon/` 框架源码里**（hybrid.py、bge.py、cached.py） | 🔴 高危 | 「独立开发」= 从 0 写，与「基于 kotaemon 改造」自相矛盾；且改了 Apache-2.0 源码却未保留修改说明/NOTICE |
| F3 | `recall@5` 98.4%→100% 来自 **125 条评测集**；但 `docs/02_retrieval_report.md` 的「三方案消融」是 **12 题**且 recall@5 三者都已 100% | 🟡 中危 | 「三方案消融」被两处用不同口径（12题 vs 125题）描述，口径不一会被追问混淆 |

**结论先行**：你的代码贡献是真实且可观的（向 kotaemon 贡献了 BM25 混合检索、bge 重排、双层磁盘缓存模块 + 独立写了 FastAPI 服务/评测体系/Docker）。问题不在"有没有做"，而在**措辞把"改造"说成了"独立从0开发"**，以及**评测证据链不完整**。下面逐条修。

---

## 1. 真实性与贡献边界（对应 proplem 第 1 节）

| 问题 | 根因/风险 | 解决方案 | 落地动作 |
|---|---|---|---|
| 1.「独立开发」vs「基于 kotaemon 改造」 | 措辞歧义，易被判定夸大 | 改为「**独立负责 / 基于 kotaemon 改造**」，并明确"向框架贡献了 X/Y/Z 模块，而非从零造框架" | 改简历行：`2026.08–2026.10 ｜ 独立负责（基于 kotaemon 改造）` |
| 2. 代码贡献占比、Git 证明 | 当前项目在本地，无 commit 历史 | 把改动拆成可证明的提交：框架增强 / 服务化 / 评测 / 部署 四组 | 见文末「行动清单 A1」建 Gitee 仓库 + 分模块 commit |
| 3. Apache-2.0 合规：许可证/NOTICE/修改说明 | 改了 `libs/kotaemon` 源码但未加修改声明 | 在仓库根加 `NOTICE` + `MODIFICATIONS.md` 说明改了哪些文件、为何改 | 见文末 A2 |
| 4. 同时做两个项目的投入时间证明 | 周期重叠（08–10 月两个项目） | 用 commit 时间戳 + 模块清单证明并行推进；面试主动讲"TradingAgents 是框架增强、本项目是 RAG 底座，技术栈互补" | 维护一份 `docs/CHANGELOG.md` 按周记录 |

**贡献边界标准话术（背下来）**：
> "我没有从零造 RAG 框架，而是基于 Apache-2.0 的 kotaemon 做改造：向框架**新增**了纯 Python BM25（`OkapiBM25`，中文 char-bigram 分词）、`HybridFusionRetriever`（RRF 融合）、`BgeReranking` 重排、以及 `CachedEmbeddings`/`CachedLLM` 双层磁盘缓存；并在其上**独立搭建**了 FastAPI 服务化、125 条量化评测体系与 Docker 部署。检索/生成主链路复用框架公开接口，未破坏原能力。"

---

## 2. 与 Agent 岗位匹配度（对应 proplem 第 2 节）

| 问题 | 根因/风险 | 解决方案 | 落地动作 |
|---|---|---|---|
| 5. RAG 不是完整 Agent，缺工具调用/Agent Loop/记忆 | 项目 1 本质是"检索+生成"，没有决策循环 | **增量**加一层 `TrafficAgent` 编排：用 ReAct/工具调用决定「检索 / 查规章表 / 计算器 / 拒答」，带多轮记忆与引用溯源（不重写 RAG，只包一层） | 见文末 A3「Agent 外壳」方案 |
| 6. 如何证明具备 Agent 开发能力 | 简历只有 TradingAgents 是多 Agent，项目 1 不是 | 双项目协同叙事：项目 1 = RAG 检索/生成底座；TradingAgents = 多智能体编排；两者共同构成"RAG + Multi-Agent"能力面 | 简历把两个项目统称「Agent / RAG 技术栈」，项目 1 强调"可升级为 Agent" |
| 7. 能否升级成 Agent？设计过吗？ | 目前没有设计文档 | 写一份 `docs/05_agent_design.md`：工具清单、Agent Loop、记忆机制、拒答策略、与现有 RAG 服务的调用关系 | 见 A3 |

**增量 Agent 外壳（不重写，只包一层）设计要点**：
- 工具：`retrieve(question)`（复用 `TrafficQAService`）、`lookup_regulation(keyword)`、`calculate(expr)`、`reject()`
- Loop：LLM 输出 `<tool>...</tool>` → 执行 → 回填 → 再决策，最多 N 轮
- 记忆：把历史 Q/A 存 `session_id` 对应磁盘/Redis，下一轮带入
- 这能直接回答"RAG 不是 Agent"的质疑，且符合你"只做增量增强"的硬约束。

---

## 3. 检索与评测技术追问（对应 proplem 第 3 节）

| 问题 | 根因/风险 | 解决方案（可直接当答案） | 落地动作 |
|---|---|---|---|
| 8. 中文分词怎么做的 | BM25 用自实现 `OkapiBM25`，**char-bigram** 分词（非 jieba） | 答："纯 Python 自实现，char-bigram 规避 jieba 词表外专业词（CBTC/ATP）切词错误，零外部依赖" | 已在 `libs/kotaemon/.../hybrid.py` |
| 9. 向量模型 / 向量库 | fastembed 本地 `bge-small-zh`；docstore=LanceDB，vectorstore=Chroma | 答："bge-small-zh 本地离线；Chroma 存向量、LanceDB 存文档（曾试 LanceDB 向量但 schema 不兼容，降级）" | 已在 `flowsettings.py` |
| 10. RRF 的 k 值/权重/候选/去重 | k=60（默认），按 rank 融合不加权，候选=两路各 top_k=5 共 10，按 doc id 去重 | 答："RRF `score=Σ1/(k+rank)` 只取排名不取分数，k=60，免归一化免调参；候选 10 条经 bge 重排取 top5" | 见 `docs/02_retrieval_report.md` |
| 11. 为什么用 RRF 不用加权 | 加权需分数归一化且对 α 敏感 | 答："BM25 分与余弦分数量纲不同，归一化策略敏感；RRF 只看排名，鲁棒无调参；保留 `alpha` 加权路径可配置消融" | 已在 `HybridFusionRetriever` |
| 12. rerank 候选/延迟/成本 | 候选 10→5；延迟 35.3ms→3.77s；仅本地算力无额外 token | 答："两阶段：先召回 10 再重排取 5；以 ~3.7s 换 MRR@5 0.926→0.955，生产可按 `use_rerank` 动态开关" | 见报告 §3/§4 |
| 13. 125 评测集构建/来源/标注 | 手工从 19 份原文抽，非 LLM 生成 | 答："single 65 / numeric 26 / table 11 / multi_hop 23；ground_truth 逐条手工抽取并带 `source_file`+`source_page` 可回溯；multi_hop 专设计跨文档推理" | 已见 `eval/results/report.md` |
| 14. recall@5 仅 +1.6% 是否过拟合 | 基线已 98.4%（19 份小语料区分度高） | 答："绝对值提升小是因为纯向量基线已很强；**真正价值在 MRR@5 0.893→0.955 与鲁棒性**（同义词/术语混淆场景，如 CBTC/ATP 案例）；小语料不证明泛化，已如实说明" | 主动讲案例 1 |
| 15. MRR@5 显著性/置信区间 | 仅给点估计 | 给 bootstrap 95% CI | 见 A4 脚本 |
| 16. LLM-as-judge 模型/prompt/一致性 | **未公开 judge 模型与 prompt**（F1 关联） | 公开 judge 模型名 + prompt 模板；对 20 条做人工双评算 Cohen's κ | 见 A4 |
| 17. faithfulness 0.800 是否偏低 | 0.8 属 RAG 正常区间；低谷来自检索未命中（q077/q110） | 答："0.8 是合理水平；低谷集中在检索 miss 的 2 题，已用'检索为空则拒答兜底'控制幻觉" | 见报告 §3 |

**⚠️ F1 必做**：让 `eval.py` 把 judge 分数写回同一份 CSV/JSON（目前全 0），否则所有生成质量数字不可复现。

---

## 4. 工程化与量化可信度（对应 proplem 第 4 节）

| 问题 | 根因/风险 | 解决方案 | 落地动作 |
|---|---|---|---|
| 18. 缓存 key 设计 / 语义不同能否命中 | key=内容 sha256；语义同表述不同**不命中** | 答："embedding key=`sha256(text)`，LLM key=`sha256(序列化messages)`；语义等价不同表述属不同 key（已知限制，生产可加规范化/语义哈希）" | 已在 `cached.py` |
| 19. 规章更新后缓存失效 | 仅 TTL，无版本号 | 答："LLM TTL 3600s + 入库时可清缓存；生产级应在 key 内纳入文档版本号" | 已在 `service.py` |
| 20.「token 成本降低 100%」不严谨 | 仅热缓存命中时为 0（20 题复跑命中率 100%） | 改为"**热缓存命中时 token 成本归零**（基准命中率 100%，单题最高加速 10.9x）" | 改简历 bullet |
| 21.「延迟降低 88.6%」基线 | 基线=冷启动 18.9s，热缓存 2.1s | 改为"平均响应延迟 18.9s→2.1s（**冷启动→热缓存，降 88.6%**）" | 改简历 bullet |
| 22. FastAPI 接口/鉴权/限流 | 无鉴权、无限流、`/ingest` 收本地路径（有风险） | 答："当前为内网 demo，未加鉴权；生产需 API Key 头 + 限流 + `/ingest` 改 multipart 上传并校验后缀" | 见 A5 加固补丁 |
| 23. pytest 只 3 passed 覆盖什么 | 仅 healthz/结构/空问题拒答，无检索/缓存/异常测试 | 补：`test_cache_hit_zero_token` / `test_retrieval_recall_subset` / `test_rerank_latency` / `test_llm_retry_fallback` | 见 A5 |
| 24. Docker 包含哪些服务 | API + 数据卷 + .env 注入 key | 答："Dockerfile.api(python:3.11-slim+uv) + compose(API:8000+卷挂载+.env)；数据持久化到 `/app/ktem_app_data`" | 需实跑 `docker build` 验证一次 |
| 25. JSON 日志字段 / 是否接 LangFuse | 用 stdlib logging+JSON（structlog 降级），**未接 LangFuse** | 答："字段 ts/level/trace_id/stage/latency_ms/n_retrieved/prompt_tokens/completion_tokens/cache_hit；trace_id 贯穿全链路；LangFuse 为后续（接口已留）" | 见 `structured_log.py` |

---

## 5. 合规与可复现（对应 proplem 第 5 节）

| 问题 | 根因/风险 | 解决方案 | 落地动作 |
|---|---|---|---|
| 26. 地铁运维数据是否涉密 | 若用内部运营数据则有合规风险 | 只用**公开**规章/规范（设计规范、维修规程类国标/行标），加 `DATA_SOURCE.md` 声明来源与公开性 | 见 A6 |
| 27. Gitee 仓库/README/架构图/一键运行/评测报告 | 当前全在本地 | 建公开仓库：README（含架构图）+ docker 一键 + 评测报告链接 + 修改说明 | 见 A1/A2 |
| 28. 真实运维人员试用/兜底/溯源/拒答/权限 | 无外部试用；已有拒答与引用溯源 | 答："面向内部 demo，未做外部用户试用；已支持引用溯源（citations 带条款号+出处）、检索为空拒答、权限控制预留" | 已在 `service.py` |

---

## 6. 简历修改对照表（改造前 → 改造后）

| 改造前（风险表述） | 改造后（稳妥表述） |
|---|---|
| `2026.08–2026.10 ｜ 独立开发` | `2026.08–2026.10 ｜ 独立负责（基于 kotaemon 改造）` |
| `token 成本降低 100%` | `热缓存命中时 token 成本归零（20 题基准命中率 100%，单题最高加速 10.9x）` |
| `平均响应延迟降低 88.6%` | `平均响应延迟 18.9s→2.1s（冷启动→热缓存，降 88.6%）` |
| `自实现 LLM-as-judge 评测生成质量（answer_relevancy 0.963、faithfulness 0.800）` | `自实现 LLM-as-judge（judge 模型：XXX，prompt 见仓库）评测四维指标，answer_relevancy 0.963、faithfulness 0.800，与人工双评一致性 κ=0.xx` |
| `实现 BM25+向量两路检索的 RRF 融合与 bge-reranker-base 两阶段重排` | `向 kotaemon 新增纯 Python BM25（char-bigram 分词）+ RRF 融合检索，并接入 bge-reranker-base 两阶段重排` |
| `实现 Embedding+LLM 双层磁盘缓存（diskcache）` | `向 kotaemon 新增 CachedEmbeddings/CachedLLM 双层磁盘缓存（diskcache，key=内容哈希）` |

---

## 7. 行动清单（按优先级）

**A 组（面试前必做，1–2 天）**
- **A1** 建 Gitee 仓库，按 4 模块提交：`feat(framework)` BM25/混合/bge/cache、`feat(service)` FastAPI、`feat(eval)` 评测、`feat(deploy)` Docker；写 README（含架构图 mermaid + 一键 `docker compose up`）。
- **A2** 加 `NOTICE` + `MODIFICATIONS.md`（列出改动的 kotaemon 文件与原因，满足 Apache-2.0）。
- **A3** 写 `docs/05_agent_design.md`，把"增量 Agent 外壳"设计落文档（即使暂不写代码，设计也能答"设计过"）。
- **A4** 修 `eval.py`：① 把 judge 分数写回 CSV/JSON（消除 F1）；② 加 bootstrap 95% CI；③ 公开 judge 模型名+prompt；④ 对 20 条做人工双评算 κ。**这一步直接决定 faithfulness/answer_relevancy 数字能不能站住。**

**B 组（加分项，3–5 天）**
- **A5** FastAPI 加固：`/query` 加 API Key 头 + 简单限流；`/ingest` 改 multipart + 后缀白名单；补 4 个 pytest（缓存零 token / 检索召回子集 / 重排延迟 / LLM 重试兜底）。
- **A6** 写 `DATA_SOURCE.md` 声明语料均为公开规章/规范，附来源列表；确认无内部涉密数据。
- 实跑 `docker build` + `docker compose up` 一次，确保部署配置真能起。

**C 组（杀手锏，可选但强烈建议）**
- 真把 `TrafficAgent` 外壳写出来（ReAct + 工具 + 多轮记忆），让项目 1 成为"带工具的 Agent"，彻底填平 RAG→Agent 的 gap，并与 TradingAgents 形成"单 Agent + 多 Agent"完整叙事。

---

## 附录：面试高频追问速答（直接背）

| 追问 | 答 |
|---|---|
| 你的 embedding / LLM 是什么 | fastembed `bge-small-zh`（本地）；LLM 走 DeepSeek OpenAI 兼容端点 |
| 检索延迟 | 混合 35.3ms，加 bge 重排 3.77s（精度换延迟，可开关） |
| 缓存命中后延迟 | 热缓存 ~2.1s，冷启动 P50 14s |
| 评测集多少条 | 125 条（single 65 / numeric 26 / table 11 / multi_hop 23），手工抽取可回溯 |
| 生成质量怎么评 | 自实现 LLM-as-judge 打 4 维（faithfulness/relevancy/precision/recall），20 条抽样 |
| 为什么不直接上 GraphRAG | 管线未跑通，降级为查询改写（更轻量零构建），如实说明反显成本意识 |
| 缓存一致性 | key=内容哈希；知识库更新靠 TTL + 可清空，生产级加文档版本号 |
| 数据合规 | 仅用公开规章/规范，附来源声明，无内部涉密数据 |
