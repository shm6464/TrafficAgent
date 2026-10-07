"""Embedding 结果缓存包装器（Phase 4 工程化）。

在任意 BaseEmbeddings 之上叠加一层磁盘缓存（diskcache），以输入文本的
内容 hash 为键缓存 embedding 向量。目的：

1. 检索链路中查询向量、文档向量可跨请求复用，避免重复调用 embedding；
2. 记录命中率，供 docs/04_engineering_report.md 量化。

设计要点：
- 仅缓存「纯文本」输入（str / Document），对 list 输入逐条缓存；
- 缓存值只存 embedding 浮点列表（JSON 可序列化），命中时重建
  DocumentWithEmbedding 返回，与底层 embedding 返回契约一致；
- 磁盘缓存默认落在 KH_APP_DATA_DIR / cache / embedding_cache，TTL 可配；
- 由 flowsettings.USE_EMBEDDING_CACHE 开关控制，关闭则直通底层，零副作用。
"""

from __future__ import annotations

import threading
from hashlib import sha256
from pathlib import Path
from typing import Optional

from theflow.settings import settings as flowsettings

from kotaemon.base import Document, DocumentWithEmbedding

from .base import BaseEmbeddings


class CachedEmbeddings(BaseEmbeddings):
    """带磁盘缓存的 embedding 包装器。

    委托给底层 BaseEmbeddings 完成真实计算，仅在命中缓存时跳过计算。
    缓存的 key = sha256(文本内容)，value = list[float]。

    Args:
        embedding: 被包装的底层 embedding 对象
        cache_dir: 磁盘缓存目录；默认 KH_APP_DATA_DIR / cache / embedding_cache
        ttl: 缓存有效期（秒），0 表示永不过期
    """

    embedding: BaseEmbeddings
    cache_dir: str = ""
    ttl: int = 0

    _cache = None
    _lock = threading.Lock()
    _hits: int = 0
    _misses: int = 0

    def _get_cache(self):
        if self._cache is None:
            import diskcache

            cache_dir = self.cache_dir or str(
                Path(getattr(flowsettings, "KH_APP_DATA_DIR", "."))
                / "cache"
                / "embedding_cache"
            )
            self._cache = diskcache.Cache(cache_dir)
        return self._cache

    @staticmethod
    def _key(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def _lookup(self, text: str) -> Optional[list[float]]:
        if not getattr(flowsettings, "USE_EMBEDDING_CACHE", True):
            return None
        cache = self._get_cache()
        key = self._key(text)
        val = cache.get(key, default=None)
        if val is not None:
            with self._lock:
                self._hits += 1
            return val
        with self._lock:
            self._misses += 1
        return None

    def _store(self, text: str, embedding: list[float]) -> None:
        if not getattr(flowsettings, "USE_EMBEDDING_CACHE", True):
            return
        cache = self._get_cache()
        ttl = self.ttl or getattr(flowsettings, "EMBEDDING_CACHE_TTL", 0)
        cache.set(self._key(text), embedding, expire=None if ttl == 0 else ttl)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total else 0.0

    def cache_stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def invoke(
        self, text, *args, **kwargs
    ) -> list[DocumentWithEmbedding]:
        input_ = self.prepare_input(text)

        # 分离命中/未命中
        cached: dict[int, list[float]] = {}
        to_compute: list[tuple[int, Document]] = []
        for idx, doc in enumerate(input_):
            content = doc.content if isinstance(doc.content, str) else str(doc.content)
            hit = self._lookup(content)
            if hit is not None:
                cached[idx] = hit
            else:
                to_compute.append((idx, doc))

        # 未命中的交给底层计算
        if to_compute:
            raw = [d for _, d in to_compute]
            results = self.embedding.run(raw, *args, **kwargs)
            for (idx, _), res in zip(to_compute, results):
                emb = list(res.embedding)
                cached[idx] = emb
                content = (
                    res.content if isinstance(res.content, str) else str(res.content)
                )
                self._store(content, emb)

        # 按原始顺序重建
        out = []
        for idx, doc in enumerate(input_):
            content = doc.content if isinstance(doc.content, str) else str(doc.content)
            out.append(
                DocumentWithEmbedding(content=content, embedding=cached[idx])
            )
        return out

    async def ainvoke(self, text, *args, **kwargs) -> list[DocumentWithEmbedding]:
        return self.invoke(text, *args, **kwargs)
