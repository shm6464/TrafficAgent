# Phase 2 检索层改造报告：BM25+向量混合检索 与 bge-reranker 两阶段重排

> 生成时间：2026-10-02
> 对应脚本：`scripts/smoke_retrieval.py`
> 原始数据：`docs/02_retrieval_results.json`

## 1. 改造内容总览

| 模块 | 文件 | 说明 |
|---|---|---|
| 纯 Python BM25 | `libs/kotaemon/kotaemon/indices/retrievers/hybrid.py` | `OkapiBM25`（自实现，无外部依赖，中文 char-bigram 分词）+ `BM25Retriever` |
| 混合检索 | 同上 | `HybridFusionRetriever`：向量 Top-K + BM25 Top-K，RRF 融合（也支持加权融合，`alpha` 可调） |
| bge-reranker | `libs/kotaemon/kotaemon/rerankings/bge.py` | `BgeReranking`，本地 sentence-transformers 加载 `BAAI/bge-reranker-base`，符合 `BaseReranking` 契约 |
| 接入管线 | `libs/ktem/ktem/index/file/pipelines.py` | `DocumentRetrievalPipeline` 增加 `hybrid_retrieval()`，由开关控制 |
| 配置开关 | `flowsettings.py` | `KH_USE_HYBRID_RETRIEVAL` / `KH_USE_RERANKING`（默认开启） |
| docstore 增强 | `libs/kotaemon/kotaemon/storages/docstores/lancedb.py` | 补 `get_all()`（BM25 语料来源） |

## 2. 检索架构（文字描述）

```
                         ┌─────────────────────────────┐
                         │        用户自然语言 query      │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              │                                                   │
     ┌────────▼────────┐                                 ┌────────▼────────┐
     │  向量检索 (Chroma)│                                 │  BM25 检索       │
     │  bge-small-zh    │                                 │  OkapiBM25       │
     │  Top-K 候选       │                                 │  Top-K 候选       │
     └────────┬────────┘                                 └────────┬────────┘
              │                                                   │
              └─────────────────────────┬─────────────────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │   RRF 融合（Reciprocal Rank │
                          │   Fusion, k=60）            │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │   bge-reranker-base 重排    │
                          │   （交叉编码器打分）          │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │   最终 Top-K 检索结果        │
                          └───────────────────────────┘
```

## 3. 消融实验结果（12 个问题，真实跑出）

### 3.1 汇总对比

| 方案 | recall@1 | recall@3 | recall@5 | 平均延迟 |
|---|---|---|---|---|
| **A 纯向量检索** | 91.7% | 91.7% | 100% | 51.5 ms |
| **B 混合检索(RRF)** | 91.7% | **100%** | 100% | 71.2 ms |
| **C 混合 + bge 重排** | 83.3% | 100% | 100% | 3935.5 ms |

### 3.2 关键结论

1. **混合检索（B）在 recall@3 上优于纯向量（A）**：91.7% → 100%。BM25 补充了向量检索遗漏的相关文档（见失败案例 1）。
2. **bge 重排（C）在 recall@1 上反而下降**：91.7% → 83.3%，且带来 3935.5ms 的显著推理开销（约 55 倍于混合检索）。
3. **本语料规模下（19 份、主题高度区分），向量检索本身已足够强**，混合检索的价值主要体现在「跨专业同名概念」这类语义混淆场景（案例 1）；bge 交叉编码器的收益在小候选集上不显著，反而可能引入噪声（案例 2）。

### 3.3 失败案例分析（3 条）

**案例 1（混合检索的收益）— Q2「CBTC ATP 层级职责」**
- 纯向量 Top1 误判为 `13_信号系统维修分级与周期.md`（"信号系统"与"CBTC"语义接近，向量被干扰）
- 混合检索 Top1 正确命中 `03_CBTC信号系统原理.md`（BM25 通过"CBTC""ATP"关键词精确匹配修正）
- **结论**：BM25 的关键词精确匹配能力弥补了向量检索在专业术语上的语义混淆。

**案例 2（重排引入噪声）— Q8「通信系统大修周期」**
- 纯向量 / 混合 Top1 均正确命中 `14_通信系统维修分级与周期.md`
- 混合+重排 Top1 却误判为 `02_设施设备运行维护管理办法_检修周期.md`（02 文档也含"通信"相关检修内容，交叉编码器认为其整体相关性更高）
- **结论**：bge 交叉编码器关注 query-doc 整体语义相关性，在"多个文档都相关"时，可能把更泛化的总则类文档排到更精确的分专业文档之前。

**案例 3（ground-truth 标注模糊性）— Q11「信号/通信维修分级异同」**
- 该问题需要同时涉及 13（信号）和 14（通信）两份文档，人工标注的单一 ground-truth（13）无法完全反映这种多文档需求
- B/C 的 Top1 命中 14 而非 13，本质是"两个都正确"，单一文档 GT 标注有局限
- **结论**：这暴露了单一 ground-truth 标注的局限，Phase 3 评测集将对 multi_hop 类问题采用"文档集合"作为 ground-truth。

## 4. 存储层选型对比（工程决策记录）

| 维度 | InMemory | Chroma | **LanceDB** | Elasticsearch |
|---|---|---|---|---|
| 部署成本 | 零 | 低（嵌入式） | 低（嵌入式） | 高（Java + 服务） |
| 持久化 | ❌ | ✅ | ✅ | ✅ |
| Windows 友好 | ✅ | ✅ | ✅ | ❌ |
| 全文检索 | ❌ | ❌ | ✅（FTS，但中文弱） | ✅（强） |
| 规模上限 | 小 | 中 | 中 | 大 |

**本项目的最终选型**：
- **docstore = LanceDB**（已采用，持久化 + Windows 友好，BM25 语料从 docstore 全量读取）
- **vectorstore = Chroma**（保持原样）

**为何 vectorstore 未切 LanceDB**（真实工程决策）：尝试切换时发现 `lancedb 0.25.x` 与 `llama-index-vector-stores-lancedb 0.1.7` 的 schema 集成层不兼容（`pyarrow.StructType.fields` 报错）；而降级 lancedb 会破坏 Phase 0/1 已建的 LanceDB docstore（违反"不破坏已有数据资产"约束）。故 docstore 用 LanceDB、vectorstore 用 Chroma，两者都满足"持久化 + 无外部服务 + Windows 友好"的目标，BM25 能力由自实现的 `OkapiBM25` 提供（绕开 lancedb FTS 的中文弱点）。

## 5. 交付物清单

| 文件 | 说明 |
|---|---|
| `libs/kotaemon/kotaemon/indices/retrievers/hybrid.py` | BM25 + 混合检索模块（新增） |
| `libs/kotaemon/kotaemon/rerankings/bge.py` | bge-reranker 重排模块（新增） |
| `libs/ktem/ktem/index/file/pipelines.py` | 检索管线接入（修改） |
| `flowsettings.py` | 开关 + bge 注册（修改） |
| `libs/kotaemon/kotaemon/storages/docstores/lancedb.py` | 补 get_all()（修改） |
| `scripts/smoke_retrieval.py` | 三方案消融对比脚本（新增） |
| `requirements.txt` | 改造新增依赖记录（新增） |
| `docs/02_retrieval_results.json` | 原始评测数据（新增） |

## 6. 复现方法

```powershell
cd D:\google\kotaemon-main\kotaemon-main
.\.venv\Scripts\python.exe scripts\smoke_retrieval.py
```

脚本会依次输出 12 个问题在三种方案下的 recall@1/3/5 与延迟，并将结果写入 `docs/02_retrieval_results.json`。
