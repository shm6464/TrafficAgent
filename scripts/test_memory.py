import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.memory_store import InMemoryStore, get_memory_store

s = InMemoryStore(max_turns=2)
for role, c in [("user","q1"),("assistant","a1"),("user","q2"),("assistant","a2"),("user","q3"),("assistant","a3")]:
    s.append("a", role, c)
h = s.get_history("a")
print("len:", len(h), "(期望 4)")
print("first:", h[0], "last:", h[-1])
assert len(h) == 4, "max_turns 裁剪失败"
assert h[0]["content"] == "q2", "应保留最近2轮"

s.clear("a")
assert s.get_history("a") == [], "clear 失败"

g = get_memory_store()
print("default backend:", type(g).__name__, "(期望 InMemoryStore)")
assert type(g).__name__ == "InMemoryStore"
print("MEMORY_OK")
