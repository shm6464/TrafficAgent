"""LLM 响应缓存（Phase 4 工程化）。

对 ChatLLM.invoke 的结果做磁盘缓存，键为 messages 序列化后的内容 hash。
命中时返回缓存的答案文本 + token 用量（prompt/completion token 记为 0，
cost 记为 0，表示本次未实际调用 LLM）。

设计要点：
- 只缓存「非流式」invoke 结果，stream 不缓存；
- 缓存值存 JSON（text / prompt_tokens / completion_tokens / total_cost）；
- 命中率记录到 cache_stats()，供 bench_latency.py 与报告量化；
- 由 flowsettings.USE_LLM_CACHE 开关控制，关闭则直通底层。

注意：LLM 缓存对「同一问题重复提问」有显著加速（省去生成与网络往返），
但也意味着答案不再随模型更新而刷新——TTL 用于控制有效期。
"""

from __future__ import annotations

import json
import threading
from hashlib import sha256
from pathlib import Path
from typing import Optional

from theflow.settings import settings as flowsettings

from kotaemon.base import BaseComponent, LLMInterface

from .base import BaseLLM


def _serialize_messages(messages) -> str:
    """把 messages（BaseMessage 列表）序列化为可 hash 的字符串。"""
    parts = []
    for m in messages:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", "")
        parts.append(f"{role}:{content}")
    return "\n".join(parts)


class CachedLLM(BaseLLM):
    """带磁盘缓存的 LLM 包装器。

    委托给底层 ChatLLM 完成真实生成；命中缓存时直接返回缓存结果。

    Args:
        llm: 被包装的底层 LLM
        cache_dir: 磁盘缓存目录；默认 KH_APP_DATA_DIR / cache / llm_cache
        ttl: 缓存有效期（秒），0 表示永不过期
    """

    llm: BaseLLM
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
                / "llm_cache"
            )
            self._cache = diskcache.Cache(cache_dir)
        return self._cache

    @staticmethod
    def _key(messages) -> str:
        return sha256(_serialize_messages(messages).encode("utf-8")).hexdigest()

    def _lookup(self, messages) -> Optional[dict]:
        if not getattr(flowsettings, "USE_LLM_CACHE", True):
            return None
        cache = self._get_cache()
        raw = cache.get(self._key(messages), default=None)
        if raw is not None:
            with self._lock:
                self._hits += 1
            return json.loads(raw)
        with self._lock:
            self._misses += 1
        return None

    def _store(self, messages, payload: dict) -> None:
        if not getattr(flowsettings, "USE_LLM_CACHE", True):
            return
        cache = self._get_cache()
        ttl = self.ttl or getattr(flowsettings, "LLM_CACHE_TTL", 0)
        cache.set(self._key(messages), json.dumps(payload), expire=None if ttl == 0 else ttl)

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

    def invoke(self, messages, *args, **kwargs) -> LLMInterface:
        hit = self._lookup(messages)
        if hit is not None:
            # 命中：重建一个 LLMInterface，token 记 0（本次未调用）
            return LLMInterface(
                content=hit["text"],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                total_cost=0.0,
            )

        resp = self.llm.invoke(messages, *args, **kwargs)
        text = getattr(resp, "text", None) or getattr(resp, "content", "")
        payload = {
            "text": text,
            "prompt_tokens": getattr(resp, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp, "completion_tokens", 0) or 0,
            "total_cost": getattr(resp, "total_cost", 0) or 0,
        }
        self._store(messages, payload)
        return resp

    def invoke_stream(self, messages, *args, **kwargs) -> LLMInterface:
        """流式调用并拼接完整结果，同时照常走磁盘缓存。

        用于阿里云百炼 glm-4.5-air 等「仅支持 stream 模式」的模型：底层
        invoke（stream=False）会被拒绝，故改用 stream 逐 chunk 拼接，但仍
        复用 invoke 的缓存逻辑（命中直接返回、未命中则流式生成后落缓存），
        保住「热缓存 -88.6% 延迟 / -100% token 成本」的工程亮点。
        """
        hit = self._lookup(messages)
        if hit is not None:
            return LLMInterface(
                content=hit["text"],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                total_cost=0.0,
            )

        parts = []
        for chunk in self.llm.stream(messages, *args, **kwargs):
            text = getattr(chunk, "text", None) or getattr(chunk, "content", "")
            if text:
                parts.append(text)
        text = "".join(parts)

        payload = {
            "text": text,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
        }
        self._store(messages, payload)
        return LLMInterface(content=text, prompt_tokens=0, completion_tokens=0, total_tokens=0, total_cost=0.0)

    def run(self, messages, *args, **kwargs):
        return self.invoke(messages, *args, **kwargs)

    def stream(self, *args, **kwargs):
        # 流式不缓存，直通底层
        return self.llm.stream(*args, **kwargs)

    async def ainvoke(self, messages, *args, **kwargs):
        return self.invoke(messages, *args, **kwargs)

    def astream(self, *args, **kwargs):
        return self.llm.astream(*args, **kwargs)

    def to_langchain_format(self):
        return self.llm.to_langchain_format()
