# Phase 4 工程化报告：缓存 / 日志 / 成本 / 服务化 / 部署

> 本报告记录交通运维知识库问答系统（kotaemon 改造）Phase 4 工程化改造的
> 实现内容与量化结果。所有数字均来自真实运行输出（`scripts/bench_latency.py`、
> `tests/test_api.py`、`eval/results/bench_latency.log`）。

---

## 1. 改造总览

| 模块 | 新增文件 | 说明 |
|---|---|---|
| 双层缓存 | `libs/kotaemon/kotaemon/embeddings/cached.py`、`libs/kotaemon/kotaemon/llms/cached.py` | Embedding 结果缓存 + LLM 响应缓存，键为内容 hash，磁盘 diskcache 后端 |
| 结构化日志 | `libs/ktem/ktem/utils/structured_log.py` | 降级方案：标准 logging + JSON formatter，输出 trace_id/阶段耗时/token/缓存命中 |
| 服务化 | `api/main.py`、`api/service.py`、`api/__init__.py` | FastAPI 服务，`/query` `/ingest` `/healthz`，复用 kotaemon 检索+生成链路 |
| 部署配置 | `Dockerfile.api`、`docker-compose.yml`、`scripts/start.ps1` | 仅写配置文件，不构建/运行容器 |
| 测试 | `tests/test_api.py` | TestClient 进程内验证接口 |
| 基准 | `scripts/bench_latency.py` | 20 题 P50/P95 + 缓存开/关对比 |

配置项（`flowsettings.py` 新增）：`USE_CACHE`、`USE_EMBEDDING_CACHE`、`USE_LLM_CACHE`、`EMBEDDING_CACHE_TTL`、`LLM_CACHE_TTL`、`USE_STRUCTLOG`、`KH_LOG_DIR`。

---

## 2. 双层缓存

### 2.1 设计

- **Embedding 缓存**（`CachedEmbeddings`）：包装任意 `BaseEmbeddings`，以输入文本的
  `sha256` 为键缓存 embedding 向量。命中时跳过本地 fastembed 计算，直接重建
  `DocumentWithEmbedding`。
- **LLM 响应缓存**（`CachedLLM`）：包装 `ChatLLM.invoke`，以 messages 序列化后的 hash
  为键缓存答案文本 + token 用量。命中时 token 记 0（本次未调用 LLM）。
- **后端**：`diskcache`（磁盘缓存，免 Redis 服务，Windows 友好），TTL 可配
  （LLM 默认 3600s，embedding 默认永久）。
- **开关**：`USE_CACHE` 总开关 + `USE_EMBEDDING_CACHE`/`USE_LLM_CACHE` 分层开关，
  关闭即直通底层，零副作用。

### 2.2 缓存收益（20 题真实基准）

| 指标 | 冷启动（无缓存） | 热缓存（命中） | 提升 |
|---|---|---|---|
| P50 延迟 | 14097.6 ms | 1899.2 ms | **-86.5%** |
| P95 延迟 | 80120.2 ms | 6951.1 ms | **-91.3%** |
| 平均延迟 | 18871.7 ms | 2143.6 ms | **-88.6%** |
| token 成本 | 0.3061 元 | 0.0000 元 | **-100%** |

- 单题加速比实测最高 **10.9x**（22.6s → 2.1s）。
- 热缓存轮 LLM 命中率 100%（20/20），embedding 命中率 100%（20/20）。
- 说明：冷启动延迟受 DeepSeek 高峰时段（14:00–18:00）限速影响，P95 高达 80s
  （含一次重试）；实际非高峰冷启动约 5–15s/条。

---

## 3. 结构化日志

- **降级说明**：`structlog` 未安装，按 INSTRUCTIONS.md 降级路径采用标准 `logging` +
  自定义 `JsonFormatter`，输出单行 JSON。
- **字段**：`ts` / `level` / `trace_id` / `stage` / `latency_ms` / `n_retrieved` /
  `prompt_tokens` / `completion_tokens` / `cache_hit`。
- **落盘**：`logs/app.log`（目录 `KH_LOG_DIR`，已加入 `.gitignore`）。
- **trace_id**：每次 `query` 调用 `reset_trace_id()` 生成新的 16 位 hex，贯穿
  检索/生成/完成的全部日志。

示例（一条 query 的结构化日志）：
```json
{"ts": "2026-10-02T18:50:00", "level": "INFO", "logger": "traffic_ops",
 "trace_id": "343f5e63e51940f8", "message": "[retrieval]",
 "latency_ms": 34.5, "n_retrieved": 5, "top_k": 5}
```

---

## 4. 成本与延迟埋点

- **埋点位置**：`api/service.py` 的 `query()` 中，用 `timed_stage` 上下文管理器
  分别记录 `retrieval` 与 `generation` 阶段耗时，`log_event` 记录 token 用量。
- **成本口径**：单价写死在 `api/service.py`（DeepSeek 空闲时段：输入 1.5 元/百万
  token、输出 4.5 元/百万 token），报告口径一致、可追溯。
- **重试容错**：LLM 生成失败（高峰超时）自动重试 3 次，仍失败则降级返回明确提示，
  不中断服务。

---

## 5. 服务化（FastAPI）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/healthz` | GET | 健康检查，返回 `{"status":"ok"}` |
| `/query` | POST | 输入 `{"question","top_k","use_rerank"}`，返回 `{"answer","citations","latency_ms","tokens","trace_id"}` |
| `/ingest` | POST | 输入 `{"file_path","reindex"}`，复用 kotaemon 入库管线 |

**测试结果**（`tests/test_api.py`，TestClient 进程内，不起服务）：

```
3 passed in 65.63s
- test_healthz PASSED
- test_query_structure PASSED   （返回 answer / citations / latency_ms / tokens / trace_id）
- test_query_reject_empty_question PASSED
```

真实 `/query` 样例（问题：信号系统的大修周期是多少年？）：
- 答案包含条款号（第十三条第（二）项）、原文引用、文档出处；
- 返回 5 条引用（`13_信号系统维修分级与周期.md` 等），带相关性得分；
- tokens: prompt=2293 / completion=713；cost=0.006648 元。

---

## 6. 部署配置（仅写文件，未构建容器）

- `Dockerfile.api`：基于 `python:3.11-slim`，`uv sync` 装依赖，`uvicorn api.main:app`
  启动，`VOLUME /app/ktem_app_data` 持久化索引。
- `docker-compose.yml`：API 服务 + 端口 8000 + 数据卷挂载 + `.env` 注入（Key 不入库）。
- `scripts/start.ps1`：本地一键启动（`.\scripts\start.ps1 [-Port 8000]`），不起 Docker。

**依据 INSTRUCTIONS.md 硬约束：本机构建/运行容器由我自行决定，此处仅提供配置。**

---

## 7. 已知限制与后续

- `structlog` 未安装，结构化日志为 logging+JSON 降级实现；如需原生 structlog 可
  `pip install structlog`（接口已兼容，`_build_logger` 会自动切换）。
- `/ingest` 接口复用 Phase 1 的入库管线，接收本地文件路径；未接入 multipart 上传
  （后续可补）。
- 冷启动延迟受 LLM 端点高峰限速影响，P95 波动大；缓存命中后可稳定在 ~2s。
