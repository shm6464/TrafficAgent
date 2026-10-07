"""验证 /agent/stream 的 token 级流式：记录 final_token 首末到达时间差。"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/agent/stream"
data = json.dumps(
    {"question": "信号系统维修分级与周期", "session_id": "s3"},
    ensure_ascii=False,
).encode("utf-8")
req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})

t0 = time.time()
final_buf = ""
first_t = None
last_t = None
with urllib.request.urlopen(req, timeout=180) as r:
    for line in r:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        evt = json.loads(line[5:].strip())
        t = evt["type"]
        dt = round(time.time() - t0, 2)
        if t in ("thought", "action", "status"):
            body = evt.get("content") or (evt.get("name", "") + " / " + evt.get("input", ""))
            print(f"[{dt:6.2f}s] {t:<11} {body}")
        elif t == "observation":
            print(f"[{dt:6.2f}s] {t:<11} {evt['content'][:50]}...")
        elif t == "final_token":
            if first_t is None:
                first_t = dt
            last_t = dt
            final_buf += evt["content"]
        elif t == "done":
            print(f"[{dt:6.2f}s] {t:<11} latency={evt['latency_ms']}ms")

print("\n=== final 答案 token 级流式 ===")
print(f"首个 token 到达 {first_t}s，末个 {last_t}s，跨度 {round((last_t - first_t) if first_t else 0, 2)}s")
print(f"final 总长 {len(final_buf)} 字")
print(f"预览: {final_buf[:100]}...")
