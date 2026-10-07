# TrafficAgent 多轮记忆（Redis）设计评审与实施建议

> 背景：同事已完成 ReAct 主循环（`api/agent.py`，162 行，质量扎实），下一步计划实现「Redis 多轮记忆」。本文给出对 Redis 方案的评审结论、必须规避的隐患，以及一处高于 Redis 本身的优先建议。
>
> 适用约束：只做增量增强，不重写已有 RAG / 检索 / 缓存逻辑；默认行为不变，新基础设施可配置开关。

---

## 1. 当前代码现状（核实）

- `api/agent.py` 已实现：`kb_search` 工具、`REACT_TEMPLATE`、手写 `TrafficAgent`（`_call_llm` / `_parse_action` / `_strip_after_observation` / `run` / `chat`）。
  - `_parse_action` 优先取 `Action` 而非 `Final Answer`，`_strip_after_observation` 截断模型脑补的 Observation —— 这两点正确治了 DeepSeek 一次性编完整 ReAct 循环的毛病，`max_iter` 兜底也到位。**ReAct 这层没问题。**
  - 多轮记忆当前是进程内 `SESSIONS: dict[str, list[dict]]`（第 14 行）+ `_history_block` 拼接（第 17–22 行）。
- `libs/kotaemon/kotaemon/llms/cached.py`：现有 LLM 缓存使用 **diskcache（本地磁盘）**，**全项目尚无 Redis**；`docker-compose.yml` 仅含 `api` 服务，无 redis 服务。
- 结论：引入 Redis = **新增一项基础设施**（redis 服务 + 连接配置 + 异常处理），不是复用已有组件。

---

## 2. 对「Redis 多轮记忆」的结论

| 维度 | 判断 |
|---|---|
| 当前你跑的方式（单进程 uvicorn） | 内存 `dict` 完全够用，Redis 此刻**非必需** |
| 真正该上 Redis 的理由 | ① 多 worker（`uvicorn --workers N`）时内存字典不共享 ② 进程重启会话丢失 ③ TTL 自动过期清理垃圾会话 |
| 项目现状 | 尚无 Redis；docker-compose 无 redis 服务 → 引入 = 新增基础设施 |
| 建议 | **做，但必须套一层接口抽象；默认走内存，Redis 可配置切换** —— 不破坏「只做增量」，也更像生产做法 |

一句话：**Redis 是「生产加固」不是「本地必需」。把它做成可插拔后端，比硬塞一个 Redis 进去价值高得多，也是更好的面试谈资。**

---

## 3. 两个比「用什么存储」更关键的隐患

**隐患 1：历史无上限。** `_history_block` 把所有轮次全拼进 prompt。一旦上 Redis 且不加 `max_turns` 上限，长会话会无限膨胀 → token 成本与延迟**持续爬升**。这是最可能变成线上 bug 的点。

**隐患 2：存储与调用耦合。** 现在 `SESSIONS` 是全局变量散在 `chat` 里。若直接换成 `redis_client.lpush(...)`，基础设施被焊死在业务代码里，Redis 一挂整个 `/agent` 就挂。

---

## 4. 推荐设计：MemoryStore 接口抽象（新增 `api/memory_store.py`）

定义接口，内存版与 Redis 版都实现它；`agent.py` 只依赖接口；**默认内存、配置切换 Redis、Redis 不可用优雅降级**。

```python
# api/memory_store.py  —— 新增文件，不改动任何现有代码
from __future__ import annotations
import os
from abc import ABC, abstractmethod


class MemoryStore(ABC):
    @abstractmethod
    def get_history(self, session_id: str) -> list[dict]: ...
    @abstractmethod
    def append(self, session_id: str, role: str, content: str) -> None: ...
    @abstractmethod
    def clear(self, session_id: str) -> None: ...


class InMemoryStore(MemoryStore):
    """开发/默认后端：零依赖，保留最近 max_turns 轮。"""
    def __init__(self, max_turns: int = 10):
        self._data: dict[str, list[dict]] = {}
        self.max_turns = max_turns

    def get_history(self, sid: str) -> list[dict]:
        return self._data.get(sid, [])

    def append(self, sid: str, role: str, content: str) -> None:
        h = self._data.setdefault(sid, [])
        h.append({"role": role, "content": content})
        cap = self.max_turns * 2
        if len(h) > cap:
            del h[: len(h) - cap]   # 只留最近 N 轮，控制 prompt 体积

    def clear(self, sid: str) -> None:
        self._data.pop(sid, None)


class RedisMemoryStore(MemoryStore):
    """生产后端：多 worker 共享 + TTL 自动过期。"""
    def __init__(self, url: str, ttl: int = 1800, max_turns: int = 10):
        import redis
        self.r = redis.Redis.from_url(url, socket_connect_timeout=3)
        self.ttl, self.max_turns = ttl, max_turns

    def append(self, sid: str, role: str, content: str) -> None:
        self.r.rpush(sid, f"{role}:{content}")
        self.r.expire(sid, self.ttl)              # 关键：会话自动过期
        self.r.ltrim(sid, -self.max_turns * 2, -1)  # 最近 N 轮

    def get_history(self, sid: str) -> list[dict]:
        return [self._parse(x) for x in self.r.lrange(sid, 0, -1)]

    def clear(self, sid: str) -> None:
        self.r.delete(sid)

    @staticmethod
    def _parse(item: bytes) -> dict:
        text = item.decode("utf-8")
        role, _, content = text.partition(":")
        return {"role": role, "content": content}


def get_memory_store() -> MemoryStore:
    """工厂：靠环境变量切换后端。"""
    if os.getenv("MEMORY_BACKEND") == "redis":
        try:
            return RedisMemoryStore(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                ttl=int(os.getenv("MEMORY_TTL", 1800)),
                max_turns=int(os.getenv("MEMORY_MAX_TURNS", 10)),
            )
        except Exception as e:  # Redis 不可用 → 降级内存，保证 /agent 不崩
            print(f"[memory] Redis 不可用，降级为内存存储: {e}")
            return InMemoryStore(max_turns=int(os.getenv("MEMORY_MAX_TURNS", 10)))
    return InMemoryStore(max_turns=int(os.getenv("MEMORY_MAX_TURNS", 10)))
```

**`agent.py` 接入方式（仅替换记忆相关 4 行，不动 ReAct 逻辑）：**

```python
from .memory_store import get_memory_store
_store = get_memory_store()          # 替换全局 SESSIONS

def _history_block(session_id: str) -> str:
    hist = _store.get_history(session_id)
    if not hist:
        return ""
    lines = [f"{'用户' if m['role']=='user' else '助手'}: {m['content']}" for m in hist]
    return "历史对话：\n" + "\n".join(lines) + "\n\n"

# 在 chat() 内：
#   _store.append(session_id, "user", question)
#   _store.append(session_id, "assistant", res["answer"])
```

> 配套 `.env` / docker-compose：加 `MEMORY_BACKEND`、`REDIS_URL`、`MEMORY_TTL`、`MEMORY_MAX_TURNS`，并在 compose 里新增可选 `redis` 服务（仅生产启用）。

---

## 5. 其他意见（按简历含金量排序）

1. **【最高优先级】把「单一工具」拆成 2–3 个工具。** 当前只有 `knowledge_base`，ReAct 退化成「RAG 多跑一轮」，辨识度低。建议拆为 `search_regulations`（规章）/ `search_cases`（案例）/ `search_equipment`（设备参数），让 planner **必须选择工具**。这是「Agent 区别于 RAG」最硬的证据，也给你多工具选择的消融素材。**比 Redis 更值钱。**
2. **【高优先级】补 Agent 自身的评测。** 已有 RAG 的 recall@5、faithfulness 消融，但**还没证明「Agent 比 RAG 好」**。建议造一个多跳/多轮小评测集（例：「先查某信号设备参数，再问对应故障处置时限」），对比 Agent vs 纯 RAG 答案质量 —— 这是简历上「引入 Agent 带来 X% 提升」的出处。（同时补上评测 CSV 中 judge 分数未写回的 F1 问题。）
3. **【中】可观测性。** `run()` 已返回 `trace`，很好。建议 `/agent` 响应再加 `retrieval_count`（本轮检索次数），面试可一句话展示「Agent 做了 N 次工具调用」。
4. **【话术】Redis 表述要准确。** 建议写：「基于 Redis 的多轮会话状态存储，支持 TTL 自动过期与多 worker 共享；通过 MemoryStore 接口抽象，开发态默认内存、生产态可切换」，别写「用 Redis 实现记忆」这种含糊说法。

---

## 6. 实施优先级建议

| 顺序 | 事项 | 理由 |
|---|---|---|
| 1 | 工具拆分（多工具） | 简历辨识度最高，体现真正的 tool-use reasoning |
| 2 | Agent vs RAG 评测 | 给「Agent 有价值」提供量化证据 |
| 3 | Redis 多轮记忆（本文方案） | 生产加固，按接口抽象 + 默认内存 + `max_turns` + TTL 落地 |
| 4 | 可观测性增强（retrieval_count 等） | 收尾打磨 |

---

## 7. 验收要点（Definition of Done）

- [ ] `MEMORY_BACKEND` 缺省时行为与原内存字典一致（回归通过）。
- [ ] 设 `MEMORY_BACKEND=redis` 后，多轮指代问题（「那它的时限是几天？」）跨请求可答。
- [ ] 长会话 prompt 长度受 `max_turns` 约束，不无限增长。
- [ ] Redis 宕机时自动降级内存，不返回 500。
- [ ] docker-compose 中 redis 为可选服务，不影响本地 `uvicorn` 直跑。
