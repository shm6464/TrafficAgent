"""交通运维知识库问答核心服务（Phase 4）。

复用 kotaemon 的检索（HybridFusionRetriever）与生成（ChatLLM + 领域 prompt）
链路，封装为可供 FastAPI 与 bench_latency.py 直接调用的无状态服务。

职责：
1. 检索：向量+BM25 混合（RRF）+ 可选 bge 重排；
2. 生成：LLM + 领域 prompt（条款号+原文引用+出处）；
3. 埋点：trace_id / 阶段耗时 / 检索命中数 / token 用量 / 缓存命中；
4. 双层缓存：LLM 响应缓存（diskCache），embedding 缓存由 CachedEmbeddings 承接。

注意：本模块是「新增模块」，不修改原 pipeline 源码，检索/生成逻辑均
通过公开接口复用，满足 INSTRUCTIONS.md「不破坏现有能力」的硬约束。
"""

from __future__ import annotations

import time
from typing import Optional

from theflow.settings import settings as flowsettings

from ktem.components import get_docstore, get_vectorstore
from ktem.embeddings.manager import embedding_models_manager
from ktem.llms.manager import llms
from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT, TRAFFIC_SYSTEM_PROMPT
from ktem.utils.structured_log import log_event, reset_trace_id, timed_stage

from kotaemon.base import HumanMessage, RetrievedDocument, SystemMessage
from kotaemon.embeddings import CachedEmbeddings
from kotaemon.indices.retrievers.hybrid import HybridFusionRetriever
from kotaemon.llms import CachedLLM


class LLMInterfaceFallback:
    """生成失败时的兜底对象，仅承载 text 与 token 字段（0）。"""

    def __init__(self, text: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text = text
        self.content = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.total_cost = 0.0


# 成本折算单价（元/百万 token，DeepSeek 空闲时段价，写死便于报告口径一致）
PRICE_PROMPT_PER_M = 1.5
PRICE_COMPLETION_PER_M = 4.5


class TrafficQAService:
    """交通运维问答服务。一次实例化后可复用（缓存跨请求生效）。"""

    def __init__(
        self,
        collection: str = "index_4",
        use_rerank: bool = True,
        top_k: int = 5,
    ):
        self.collection = collection
        self.use_rerank = use_rerank
        self.top_k = top_k

        # embedding：叠加磁盘缓存（由 USE_EMBEDDING_CACHE 控制）
        base_emb = embedding_models_manager.get_default()
        self.embedding = CachedEmbeddings(embedding=base_emb)

        self.vector_store = get_vectorstore(collection)
        self.doc_store = get_docstore(collection)

        # 检索器：默认开启 bge 重排时挂载本地 bge-reranker。
        # 直接实例化 BgeReranking（Phase 2 实现），不依赖 manager 的 DB 状态
        # （manager 中可能只注册了未配置 key 的 cohere，走 get_default 会 403）。
        rerankers = []
        if use_rerank:
            try:
                from kotaemon.rerankings import BgeReranking

                rerankers = [BgeReranking(model_name="BAAI/bge-reranker-base")]
            except Exception:
                rerankers = []

        self.retriever = HybridFusionRetriever(
            embedding=self.embedding,
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            top_k=top_k,
            fusion_method="rrf",
            rerankers=rerankers,
        )

        # LLM：叠加响应缓存（由 USE_LLM_CACHE 控制）
        base_llm = llms.get_default()
        self.llm = CachedLLM(llm=base_llm)

    def _build_prompt(self, question: str, evidence: str) -> str:
        return TRAFFIC_QA_TEXT_PROMPT.format(
            lang="中文", context=evidence, question=question
        )

    def _format_evidence(self, docs: list[RetrievedDocument]) -> str:
        parts = []
        for i, d in enumerate(docs, 1):
            fn = d.metadata.get("file_name", "-")
            parts.append(f"[出处{i}: {fn}]\n{d.text}")
        return "\n\n".join(parts)

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """执行一次完整问答。返回结构化结果（供 API 序列化）。"""
        trace_id = reset_trace_id()
        k = top_k or self.top_k
        t_start = time.time()

        # 1. 检索
        with timed_stage("retrieval") as st:
            docs = self.retriever.run(question, top_k=k)
            st.extra(n_retrieved=len(docs), top_k=k)

        # 2. 生成（带重试容错：DeepSeek 高峰时段偶发超时）
        evidence = self._format_evidence(docs)
        prompt = self._build_prompt(question, evidence)
        messages = [SystemMessage(content=TRAFFIC_SYSTEM_PROMPT), HumanMessage(content=prompt)]

        resp = None
        last_err = None
        with timed_stage("generation") as st:
            for attempt in range(1, 4):
                try:
                    # 用 invoke_stream：兼容 glm-4.5-air 等「仅支持 stream」的模型，
                    # 同时保留 CachedLLM 的磁盘缓存（非流式 invoke 会被百炼拒绝）。
                    resp = self.llm.invoke_stream(messages)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    print(f"[LLM 重试 {attempt}/3] {type(e).__name__}: {e}")
                    time.sleep(2 * attempt)
            if resp is None:
                # 重试 3 次仍失败：降级为明确提示，不中断服务（高峰期限速兜底）
                resp = LLMInterfaceFallback(
                    text=f"（生成失败，请稍后重试。原因：{type(last_err).__name__}）",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            st.extra(
                prompt_tokens=getattr(resp, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(resp, "completion_tokens", 0) or 0,
            )

        answer_text = getattr(resp, "text", None) or getattr(resp, "content", "")
        prompt_tokens = getattr(resp, "prompt_tokens", 0) or 0
        completion_tokens = getattr(resp, "completion_tokens", 0) or 0
        total_cost = getattr(resp, "total_cost", 0) or 0

        # 成本折算（若底层未给 total_cost，按单价自算）
        if total_cost == 0 and (prompt_tokens or completion_tokens):
            total_cost = (
                prompt_tokens / 1e6 * PRICE_PROMPT_PER_M
                + completion_tokens / 1e6 * PRICE_COMPLETION_PER_M
            )

        # 3. 引用组装
        citations = []
        for d in docs:
            citations.append(
                {
                    "doc": d.metadata.get("file_name", "-"),
                    "page": d.metadata.get("page_label", None),
                    "score": round(float(d.score), 4) if d.score is not None else None,
                    "text": (d.text or "")[:200],
                }
            )

        total_ms = round((time.time() - t_start) * 1000, 2)

        result = {
            "trace_id": trace_id,
            "answer": answer_text,
            "citations": citations,
            "latency_ms": total_ms,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            "cost_rmb": round(total_cost, 6),
            "cache_stats": {
                "llm": self.llm.cache_stats() if hasattr(self.llm, "cache_stats") else {},
                "embedding": (
                    self.embedding.cache_stats()
                    if hasattr(self.embedding, "cache_stats")
                    else {}
                ),
            },
        }
        log_event("query_done", latency_ms=total_ms, n_citations=len(citations))
        return result

    def cache_stats(self) -> dict:
        return {
            "llm": self.llm.cache_stats() if hasattr(self.llm, "cache_stats") else {},
            "embedding": (
                self.embedding.cache_stats()
                if hasattr(self.embedding, "cache_stats")
                else {}
            ),
        }


# 进程级单例（懒加载，避免 import 时触发模型加载）
_service: Optional[TrafficQAService] = None


def get_service(**kwargs) -> TrafficQAService:
    global _service
    if _service is None:
        _service = TrafficQAService(**kwargs)
    return _service
