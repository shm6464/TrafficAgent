# api/memory_store.py
"""多轮会话记忆存储抽象层（可插拔后端）。

设计目标：
- 默认内存（零依赖，本地开发即用）；
- 设 MEMORY_BACKEND=redis 时切换 Redis（多 worker 共享 + TTL 自动过期）；
- Redis 不可用时优雅降级为内存，保证 /agent 不因基础设施故障而 500。

硬约束：只新增本文件，不改动 libs/kotaemon/ 与 api/service.py。
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


class MemoryStore(ABC):
    @abstractmethod
    def get_history(self, session_id: str) -> list[dict]:
        """返回该会话的历史消息（[{"role","content"}, ...]）。"""

    @abstractmethod
    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息，并自动裁剪到最近 max_turns 轮。"""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """删除整个会话。"""


class InMemoryStore(MemoryStore):
    """开发/默认后端：进程内字典，保留最近 max_turns 轮。"""

    def __init__(self, max_turns: int = 10):
        self._data: dict[str, list[dict]] = {}
        self.max_turns = max_turns

    def get_history(self, session_id: str) -> list[dict]:
        return list(self._data.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        h = self._data.setdefault(session_id, [])
        h.append({"role": role, "content": content})
        cap = self.max_turns * 2
        if len(h) > cap:
            del h[: len(h) - cap]          # 只留最近 N 轮，控制 prompt 体积

    def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisMemoryStore(MemoryStore):
    """生产后端：多 worker 共享 + TTL 自动过期。消息用 JSON 序列化避免分隔符歧义。"""

    def __init__(self, url: str, ttl: int = 1800, max_turns: int = 10):
        import redis
        self.r = redis.Redis.from_url(url, socket_connect_timeout=3)
        self.ttl = ttl
        self.max_turns = max_turns

    def append(self, session_id: str, role: str, content: str) -> None:
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        self.r.rpush(session_id, payload)
        self.r.expire(session_id, self.ttl)              # 会话自动过期
        self.r.ltrim(session_id, -self.max_turns * 2, -1)  # 最近 N 轮

    def get_history(self, session_id: str) -> list[dict]:
        out = []
        for item in self.r.lrange(session_id, 0, -1):
            try:
                out.append(json.loads(item.decode("utf-8")))
            except Exception:
                continue
        return out

    def clear(self, session_id: str) -> None:
        self.r.delete(session_id)


def get_memory_store() -> MemoryStore:
    """工厂：靠环境变量 MEMORY_BACKEND 切换后端；Redis 不可用自动降级内存。"""
    max_turns = int(os.getenv("MEMORY_MAX_TURNS", "10"))
    if os.getenv("MEMORY_BACKEND") == "redis":
        try:
            return RedisMemoryStore(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                ttl=int(os.getenv("MEMORY_TTL", "1800")),
                max_turns=max_turns,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[memory] Redis 不可用，降级为内存存储: {e}")
            return InMemoryStore(max_turns=max_turns)
    return InMemoryStore(max_turns=max_turns)
