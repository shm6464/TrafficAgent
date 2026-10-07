"""端到端验证 /agent：多轮记忆 + 指代追问。复用即可（显式 UTF-8，规避中文编码坑）。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path, payload, timeout=180):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# 等 healthz 就绪（服务启动 + 首次懒加载模型可能较慢）
for _ in range(120):
    try:
        urllib.request.urlopen(BASE + "/healthz", timeout=2)
        print("healthz OK")
        break
    except Exception:
        time.sleep(1)
else:
    print("服务未就绪")
    raise SystemExit(1)

# 第一问
r1 = post("/agent", {"question": "信号系统维修分级与周期", "session_id": "s1"})
print("\n=== 第一问 ===")
print("answer:", r1.get("answer", "")[:600])
print("latency_ms:", r1.get("latency_ms"))
print("trace 片段:", (r1.get("trace", "") or "")[:300])

# 第二问（指代追问，验证多轮记忆）
r2 = post("/agent", {"question": "那它的月检间隔是多久？", "session_id": "s1"})
print("\n=== 第二问（指代） ===")
print("answer:", r2.get("answer", "")[:600])
