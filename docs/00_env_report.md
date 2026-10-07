# Phase 0 环境体检报告

> 生成时间：2026-10-01 18:30（GMT+8）
> 项目：kotaemon 二次改造 —— 城市轨道交通运维知识库问答系统
> 本报告所有数据均为脚本真实跑出，未编造。

---

## 1. 环境概况

| 项 | 值 |
|---|---|
| 项目根目录 | `D:\google\kotaemon-main\kotaemon-main` |
| Python 解释器 | `.venv\Scripts\python.exe`（Python 3.10.21） |
| pip | 23.0.1（Phase 0 已修复，原 venv 缺失 pip） |
| git | 2.55.0.windows.3（Phase 0 已 `git init` 建立基线） |
| 操作系统 | Windows（PowerShell 语法执行） |

## 2. 关键包版本

| 包 | 版本 | 备注 |
|---|---|---|
| kotaemon | 0.0.1 | 本仓库 `libs/kotaemon`（非 PyPI 同名包） |
| ktem | 0.0.1 | 本仓库 `libs/ktem` |
| lancedb | 0.25.1 | ✅ 文档存储（FTS 全文检索） |
| chromadb | 0.5.16 | ✅ 向量存储 |
| fastapi | 0.112.1 | ✅ |
| fastembed | 0.8.1 | ✅ 本地 Embedding（离线缓存） |
| gradio | 4.39.0 | Web UI |
| theflow | 0.8.6 | 组件框架（BaseComponent 基类） |
| numpy | 1.26.4 | |
| pydantic | 2.10.6 | |
| langchain | 0.2.15 | |
| langchain-core | 0.2.43 | |
| langchain-openai | 0.1.25 | |
| pypdf | 4.2.0 | |
| torch | ❌ 未安装 | 本地 bge-reranker 需 Phase 2 安装 |
| sentence-transformers | ❌ 未安装 | Phase 2 bge-reranker 依赖 |
| FlagEmbedding | ❌ 未安装 | Phase 2 备选 |
| rank_bm25 | ❌ 未安装 | Phase 2 BM25 依赖 |
| ragas | ❌ 未安装 | Phase 3 评测依赖 |
| structlog | ❌ 未安装 | Phase 4 日志依赖 |

## 3. 配置项（`.env` 键名，不含值）

`.env` 中出现的配置键名（**不记录值，Key 安全**）：

```
AUTHENTICATION_METHOD, AZURE_DI_CREDENTIAL, AZURE_DI_ENDPOINT,
AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
AZURE_OPENAI_ENDPOINT, COHERE_API_KEY, FASTEMBED_CACHE_PATH,
GRAPHRAG_API_KEY, GRAPHRAG_EMBEDDING_MODEL, GRAPHRAG_LLM_MODEL,
HF_ENDPOINT, HF_HUB_DISABLE_XET, HF_HUB_OFFLINE,
KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET, KEYCLOAK_REALM, KEYCLOAK_SERVER_URL,
LOCAL_MODEL, LOCAL_MODEL_EMBEDDINGS, OPENAI_API_BASE, OPENAI_API_KEY,
OPENAI_API_VERSION, OPENAI_CHAT_MODEL, OPENAI_EMBEDDINGS_MODEL,
PADDLE_DEVICE, PDFJS_VERSION_DIST, PDF_SERVICES_CLIENT_ID, PDF_SERVICES_CLIENT_SECRET,
USE_CUSTOMIZED_GRAPHRAG_SETTING, VOYAGE_API_KEY
```

## 4. 当前生效的存储 / 模型配置

| 配置项 | 当前生效值 |
|---|---|
| `KH_DOCSTORE` | `kotaemon.storages.LanceDBDocumentStore`（持久化，`user_data/docstore`） |
| `KH_VECTORSTORE` | `kotaemon.storages.ChromaVectorStore`（持久化，`user_data/vectorstore`） |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5`（FastEmbed，本地离线缓存，`default=True`） |
| LLM 模型 | `deepseek-v4-flash`（OpenAI 兼容端点 `https://api.deepseek.com`，`default=True`） |
| GraphRAG 开关 | MS GraphRAG ✅、LightRAG ✅、NanoGraphRAG ❌ |

## 5. 已入库数据资产（改造前基线）

| 项 | 值 |
|---|---|
| File Index | `index_1` |
| 文档 | `智能客服工单知识库.pdf`（48134 字节，13,022 token） |
| chunk 数 | 86 |
| 解析器 | `PDFThumbnailReader` |
| docstore | `user_data/docstore/index_1.lance`（LanceDB） |
| vectorstore | `user_data/vectorstore/`（Chroma） |

> 注：该 PDF 是「智能客服工单」主题，Phase 0 仅用于验证链路可用；Phase 1 将新增 `traffic_ops_kb` 索引承接交通语料，原 index_1 保留作为改造前对照。

## 6. 基线冒烟结果（脚本化，未启动任何服务）

冒烟脚本：`scripts/smoke_baseline.py`（纯脚本直调 Python API，不起 Web UI / 后端服务）。

### 6.1 检索 + 答案生成 + 出处引用（每问题 3 次取均值）

| 问题 | 平均检索延迟 | 平均首字节延迟 | 平均总延迟 | 返回出处引用 |
|---|---|---|---|---|
| 工单的优先级是如何划分的？ | 0.260s | 0.260s | 9.685s | ✅（文件名+页码） |
| 如何处理客户投诉类工单？ | 0.017s | 0.017s | 15.996s | ✅（文件名+页码） |

### 6.2 答案质量抽样（真实输出摘要）

- 问题 1 答案要点：工单优先级分 **P0 紧急 / P1 高 / P2 中 / P3 低** 四级，依据影响范围、严重程度划分（命中 P3、P4、P5、P18 页）。
- 问题 2 答案要点：投诉工单归入「投诉建议」分类并路由，按规则流转（命中 P3、P6、P14、P18 页）。

### 6.3 关键工程发现（Phase 2+ 的改造依据）

1. **混合检索 FTS 线程问题**：`VectorRetrieval` 的 `hybrid` 模式会在 `run()` 内开双线程，其中全文检索线程走 lancedb 的 FTS（英文 `en_stem` tokenizer），对中文查询在 Windows 下触发 C++ 层问题导致进程被 SIGTERM。→ Phase 2 将以**纯 Python BM25** 替代该 FTS 路径（正好是简历亮点）。
2. **theflow 框架层 `__call__` 问题**：`BaseComponent`（继承 `theflow.Function`）的 `__call__` 在 Windows 上对 embedding / TokenSplitter / LLM 组件会触发进程被杀；直接调 `.run()` / `.invoke()` 则稳定。→ 后续所有脚本化验证统一用 `.run()` / `.invoke()` 直调，规避框架层。
3. **引用生成依赖 function calling**：`AnswerWithContextPipeline` 的 `CitationPipeline` 依赖 LLM function calling（DeepSeek 端点不稳定）。→ 冒烟阶段引用口径改为「检索命中文档出处（文件名+页码）」，更真实可控；Phase 1 领域 prompt 会显式要求条款级引用。

## 7. 回滚点与进程状态

- git 基线提交：`a64575e baseline: original kotaemon`（+ `21c10b8` 忽略缓存产物）。
- 本次冒烟**未启动** Web UI / uvicorn / Gradio；端口 7860 / 8000 无占用，无遗留后台进程。
- 未改动、删除、重建任何 `ktem_app_data/` 已有数据资产。
