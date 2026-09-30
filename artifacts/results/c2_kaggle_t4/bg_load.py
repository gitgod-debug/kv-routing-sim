
import json, random, sys, threading, time, urllib.request, pathlib
port, conc, shared_n, private_n, out, stop, model = sys.argv[1:8]
conc, shared_n, private_n, out = int(conc), int(shared_n), int(private_n), int(out)
shared = [1000 + (7 * j) % 90000 for j in range(shared_n)]
done = [0]
def worker(k):
    rng = random.Random(k)
    while not pathlib.Path(stop).exists():
        ids = shared + [rng.randrange(1000, 100000) for _ in range(private_n)]
        body = {"model": model, "prompt": ids, "max_tokens": out, "temperature": 0.0,
                "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                for raw in resp:
                    line = raw.strip()
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        json.loads(line[5:].strip())
            done[0] += 1
        except Exception:
            time.sleep(0.5)
threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(conc)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(json.dumps({"completed": done[0]}))
