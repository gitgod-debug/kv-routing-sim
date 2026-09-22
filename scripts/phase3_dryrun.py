#!/usr/bin/env python3
"""Dry-run notebooks/phase3_calibration.ipynb on this machine, against a MOCK engine.

    python scripts/phase3_dryrun.py [--vllm-src PATH/TO/site-packages/vllm] [--keep]

The mock is a real HTTP server on a real port, started by the notebook through a `vllm` shim on
PATH exactly as it starts the real engine. It speaks the subset of vLLM's API the notebook uses —
/health, /metrics, streamed /v1/completions with usage, /reset_prefix_cache, /server_info — and
schedules requests IN REAL TIME with this repo's own vLLM model (`sim/replica.py`,
cache_model="vllm"). It also rejects token ids outside Qwen3's vocabulary, as vLLM does.

⚠️ What a dry-run proves, and what it cannot: the PLUMBING, end to end — fingerprints, streaming,
resets, the restart fallback, counter deltas, the comparison and the verdict. It cannot show that
vLLM behaves like the model, because here the model IS the engine. That is the GPU run's job. So:
the `vllm` model must come out EXACT here (anything else is a notebook bug), and the `lru` model
must visibly miss on W1 (else the comparison could not detect a wrong model).

`--vllm-src` points the notebook's source audit at a real vLLM package, so its needles are checked
against real code. Without it, a synthetic package holding just the needles is used.

Scenarios: A the real setup · B no reset endpoint (restart fallback) · C the override ignored
(must stop at the capacity check) · D a stale mock already on the port (must refuse to launch).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import itertools
import json
import os
import pathlib
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "phase3_calibration.ipynb"
QWEN3_VOCAB = 151_936
LE = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)  # the mock's histogram buckets; the real ones differ


# ── the mock engine ───────────────────────────────────────────────────────────────────────────


class MockEngine:
    """The simulator's vLLM replica, stepped in real time, with vLLM-style cumulative counters."""

    def __init__(self, num_blocks: int, block_size: int, max_num_seqs: int) -> None:
        sys.path.insert(0, str(REPO / "src"))
        from kv_routing_sim.sim.cost_model import CostProfile

        self.bs = block_size
        # Fast, but with a real per-sequence cost so a concurrent run queues and preempts.
        self.profile = CostProfile(
            name="mock", prefill_base_s=1e-3, prefill_per_token_s=2e-5, decode_base_s=5e-4,
            decode_per_seq_s=2.5e-3, kv_cache_blocks=num_blocks, block_size=block_size,
            max_num_seqs=max_num_seqs,
        )
        self.cv = threading.Condition()
        self.replica = self._new_replica()
        self.folded: dict[str, float] = {}  # counters of replicas retired by a reset
        self.hist: dict[str, list[float]] = {"lifetime": [], "idle": [], "gap": []}
        self.tokens = {"prompt": 0, "generation": 0, "local_compute": 0, "local_cache_hit": 0}
        self.streams: dict[str, queue.Queue] = {}
        self.emitted: dict[str, int] = {}
        self.ids = itertools.count()
        self.t0 = time.monotonic()
        threading.Thread(target=self._loop, daemon=True).start()

    def _new_replica(self):
        from kv_routing_sim.sim.replica import Replica

        return Replica(0, self.profile, cache_model="vllm", track_block_metrics=True)

    def submit(self, ids: list[int], max_tokens: int) -> tuple[str, queue.Queue]:
        from kv_routing_sim.traces.hashing import chain_hashes
        from kv_routing_sim.traces.schema import NormalizedRequest

        q: queue.Queue = queue.Queue()
        with self.cv:
            rid = f"cmpl-{next(self.ids)}"
            now = time.monotonic() - self.t0
            self.streams[rid], self.emitted[rid] = q, 0
            self.replica.enqueue(
                NormalizedRequest(rid, now, chain_hashes(ids, self.bs), len(ids), max_tokens)
            )
            self.cv.notify()
        return rid, q

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self.replica.has_work:
                    self.cv.wait()
                dur, finished = self.replica.step(time.monotonic() - self.t0)
                done = {s.request.request_id for s in finished}
                out = []
                for seq in [*self.replica.running, *finished]:
                    rid, n = seq.request.request_id, seq.tokens_emitted
                    if n > self.emitted[rid]:
                        if self.emitted[rid] == 0:  # first token: vLLM books the prompt here
                            cached = seq.matched_blocks * self.bs
                            self.tokens["prompt"] += seq.request.prompt_tokens
                            self.tokens["local_cache_hit"] += cached
                            self.tokens["local_compute"] += seq.request.prompt_tokens - cached
                        self.tokens["generation"] += n - self.emitted[rid]
                        self.emitted[rid] = n
                        usage = None
                        if rid in done:
                            usage = {
                                "prompt_tokens": seq.request.prompt_tokens,
                                "completion_tokens": n,
                                "total_tokens": seq.request.prompt_tokens + n,
                                "prompt_tokens_details": {
                                    "cached_tokens": seq.matched_blocks * self.bs
                                },
                            }
                        out.append((self.streams[rid], n, usage))
                for rid in done:
                    del self.streams[rid], self.emitted[rid]
            time.sleep(dur)  # the step takes this long; its tokens land at its end
            for q, n, usage in out:
                q.put(("token", n))
                if usage is not None:
                    q.put(("done", usage))

    def _snapshot(self, rep) -> dict[str, float]:
        p, ev = rep.pool, rep.pool.eviction_events
        return {
            "queries": p.queries, "hits": p.hits, "preemptions": rep.preemptions,
            "events": len(ev), "gaps": sum(len(e.reuse_gaps_s) for e in ev),
        }

    def reset(self) -> bool:
        with self.cv:
            if self.replica.has_work:
                return False
            for k, v in self._snapshot(self.replica).items():
                self.folded[k] = self.folded.get(k, 0) + v
            ev = self.replica.pool.eviction_events
            self.hist["lifetime"] += [e.lifetime_s for e in ev]
            self.hist["idle"] += [e.idle_s for e in ev]
            self.hist["gap"] += [g for e in ev for g in e.reuse_gaps_s]
            self.replica = self._new_replica()
            return True

    def metrics_text(self) -> str:
        with self.cv:
            cur = self._snapshot(self.replica)
            tot = {k: self.folded.get(k, 0) + v for k, v in cur.items()}
            ev = self.replica.pool.eviction_events
            hist = {
                "lifetime": self.hist["lifetime"] + [e.lifetime_s for e in ev],
                "idle": self.hist["idle"] + [e.idle_s for e in ev],
                "gap": self.hist["gap"] + [g for e in ev for g in e.reuse_gaps_s],
            }
            running, waiting = len(self.replica.running), len(self.replica.queue)
            tk = dict(self.tokens)
        lab = 'engine="0",model_name="mock"'
        rows = [
            "# HELP vllm:prefix_cache_queries Prefix cache queries, in terms of number of tokens.",
            "# TYPE vllm:prefix_cache_queries counter",
            f"vllm:prefix_cache_queries_total{{{lab}}} {float(tot['queries'])}",
            f"vllm:prefix_cache_queries_created{{{lab}}} 1.7e9",
            f"vllm:prefix_cache_hits_total{{{lab}}} {float(tot['hits'])}",
            f"vllm:num_preemptions_total{{{lab}}} {float(tot['preemptions'])}",
            (
                f'vllm:prompt_tokens_by_source_total{{{lab},source="local_compute"}} '
                f"{float(tk['local_compute'])}"
            ),
            (
                f'vllm:prompt_tokens_by_source_total{{{lab},source="local_cache_hit"}} '
                f"{float(tk['local_cache_hit'])}"
            ),
            f'vllm:prompt_tokens_by_source_total{{{lab},source="external_kv_transfer"}} 0.0',
            f"vllm:prompt_tokens_total{{{lab}}} {float(tk['prompt'])}",
            f"vllm:generation_tokens_total{{{lab}}} {float(tk['generation'])}",
            f"vllm:num_requests_running{{{lab}}} {float(running)}",
            f"vllm:num_requests_waiting{{{lab}}} {float(waiting)}",
        ]
        for fam, key in (
            ("vllm:kv_block_lifetime_seconds", "lifetime"),
            ("vllm:kv_block_idle_before_evict_seconds", "idle"),
            ("vllm:kv_block_reuse_gap_seconds", "gap"),
        ):
            vals = hist[key]
            for le in LE:
                n = sum(1 for v in vals if v <= le)
                rows.append(f'{fam}_bucket{{{lab},le="{le}"}} {float(n)}')
            rows.append(f'{fam}_bucket{{{lab},le="+Inf"}} {float(len(vals))}')
            rows.append(f"{fam}_count{{{lab}}} {float(len(vals))}")
            rows.append(f"{fam}_sum{{{lab}}} {float(sum(vals))}")
        return "\n".join(rows) + "\n"


def _handler(engine: MockEngine, no_reset: bool, blocks: int):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet, like vLLM's access log being off
            pass

        def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self._send(200, "")
            if path == "/metrics":
                return self._send(200, engine.metrics_text(), "text/plain; version=0.0.4")
            if path == "/server_info":
                cfg = {"scheduler_config": {"async_scheduling": False},
                       "cache_config": {"num_gpu_blocks_override": blocks}}
                return self._send(200, json.dumps({"vllm_config": cfg}))
            return self._send(404, '{"detail":"Not Found"}')

        def do_POST(self):
            path = self.path.split("?")[0]
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if path == "/reset_prefix_cache":
                if no_reset:
                    return self._send(404, '{"detail":"Not Found"}')
                ok = engine.reset()
                print("Successfully reset prefix cache" if ok else "Failed to reset", flush=True)
                return self._send(200, json.dumps({"success": ok}))
            if path != "/v1/completions":
                return self._send(404, '{"detail":"Not Found"}')
            req = json.loads(body)
            ids, max_tokens = req["prompt"], req["max_tokens"]
            bad = [t for t in ids if not 0 <= t < QWEN3_VOCAB]
            if bad or not req.get("ignore_eos") or not req.get("stream"):
                return self._send(400, json.dumps({"error": f"rejected: bad ids {bad[:3]}"}))
            rid, q = engine.submit(ids, max_tokens)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(text: str) -> None:
                data = text.encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            while True:
                kind, val = q.get()
                if kind == "token":
                    reason = "length" if val == max_tokens else None
                    obj = {"id": rid, "object": "text_completion", "model": "mock",
                           "choices": [{"index": 0, "text": " tok", "finish_reason": reason}],
                           "usage": None}
                    chunk(f"data: {json.dumps(obj)}\n\n")
                else:
                    chunk(f"data: {json.dumps({'id': rid, 'choices': [], 'usage': val})}\n\n")
                    chunk("data: [DONE]\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return

    return Handler


def mock_main(argv: list[str]) -> None:
    """Stands in for `vllm serve MODEL ...`. Reads only the flags it needs; ignores the rest."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-gpu-blocks-override", type=int)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--kv-cache-metrics", action="store_true")
    args, _ = ap.parse_known_args(argv)
    blocks = args.num_gpu_blocks_override or 9000
    if os.environ.get("MOCK_IGNORE_OVERRIDE"):
        blocks = 600
    print(f"INFO [kv_cache_utils.py] Overriding num_gpu_blocks=9000 with "
          f"num_gpu_blocks_override={args.num_gpu_blocks_override}", flush=True)
    print(f"INFO [kv_cache_utils.py] GPU KV cache size: {blocks * args.block_size:,} tokens, "
          f"Maximum concurrency for {args.max_model_len:,} tokens per request: "
          f"{blocks * args.block_size / args.max_model_len:.2f}x", flush=True)
    print(f"INFO [core.py] ObservabilityConfig(kv_cache_metrics={args.kv_cache_metrics}, "
          "kv_cache_metrics_sample=1.0)", flush=True)
    engine = MockEngine(blocks, args.block_size, args.max_num_seqs)
    handler = _handler(engine, bool(os.environ.get("MOCK_NO_RESET")), blocks)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True
    print(f"INFO Uvicorn running on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


# ── the harness ───────────────────────────────────────────────────────────────────────────────


def _cells(runs_per_workload: int | None) -> list[str]:
    nb = json.loads(NOTEBOOK.read_text())
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(ln for ln in src.split("\n") if not ln.lstrip().startswith("!"))
        src = src.replace("STARTUP_TIMEOUT_S = 1800", "STARTUP_TIMEOUT_S = 60")
        if runs_per_workload is not None:
            src = src.replace("RUNS_PER_WORKLOAD = 3", f"RUNS_PER_WORKLOAD = {runs_per_workload}")
        out.append(src)
    return out


def _needles(cells: list[str]) -> list[str]:
    """Every needle the audit cell looks for, read from the cell itself (CITED and USED)."""
    audit = next(c for c in cells if "CITED = [" in c)
    found = []
    for node in ast.walk(ast.parse(audit)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("CITED", "USED"):
            found += [needle for _, needle in ast.literal_eval(node.value)]
    return found


def _fake_vllm(dst: pathlib.Path, vllm_src: pathlib.Path | None, needles: list[str]) -> None:
    pkg = dst / "vllm"
    if vllm_src is not None:
        shutil.copytree(vllm_src, pkg, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    else:
        pkg.mkdir(parents=True)
        (pkg / "_needles.py").write_text("\n".join(f"# {n}" for n in needles) + "\n")
    (pkg / "__init__.py").write_text('__version__ = "0.30.0"\n')


def _shim(bin_dir: pathlib.Path) -> None:
    bin_dir.mkdir()
    shim = bin_dir / "vllm"
    shim.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"sys.path[:0] = [{str(REPO / 'scripts')!r}, {str(REPO / 'src')!r}]\n"
        "from phase3_dryrun import mock_main\nmock_main(sys.argv[1:])\n"
    )
    shim.chmod(0o755)


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def run_scenario(name: str, vllm_src, env: dict[str, str], runs: int | None, keep: bool) -> dict:
    """Execute every code cell in one namespace, the way Run-All does, and report."""
    if not _port_free(8000) and name != "D":
        raise SystemExit("port 8000 is busy — stop whatever holds it first")
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"phase3_dryrun_{name}_"))
    cells = _cells(runs)
    _fake_vllm(work / "pkg", vllm_src, _needles(cells))
    _shim(work / "bin")
    saved_env, saved_cwd, saved_path = dict(os.environ), os.getcwd(), list(sys.path)
    os.environ.update(env)
    os.environ["PATH"] = f"{work / 'bin'}{os.pathsep}{os.environ['PATH']}"
    sys.path.insert(0, str(work / "pkg"))
    sys.modules.pop("vllm", None)
    os.chdir(work)
    ns: dict = {"__name__": "__main__"}
    buf = io.StringIO()
    err, t0 = None, time.time()
    try:
        with contextlib.redirect_stdout(buf):
            for i, src in enumerate(cells):
                exec(compile(src, f"cell{i}", "exec"), ns)  # noqa: S102 - the point of a dry-run
    except Exception as e:  # noqa: BLE001 - whatever a cell raises IS the dry-run's finding
        err = f"{type(e).__name__} in code cell {i}: {e}"
    finally:
        with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()):
            ns["stop_server"]()
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
        sys.path[:] = saved_path
        sys.modules.pop("vllm", None)
    (work / "dryrun_stdout.txt").write_text(buf.getvalue())
    out = {"name": name, "error": err, "seconds": round(time.time() - t0, 1),
           "stdout": buf.getvalue(), "ns": ns, "dir": work,
           "zip_written": (work / "phase3_results.zip").exists()}
    if not keep and err is None:
        shutil.rmtree(work, ignore_errors=True)
    return out


def _check_a(r: dict) -> list[str]:
    """The real setup. The vllm model must be EXACT (it is the engine); lru must miss on W1."""
    problems = []
    ns = r["ns"]
    runs = ns.get("RUNS", [])
    if len(runs) != 3 * ns["RUNS_PER_WORKLOAD"]:
        problems.append(f"expected {3 * ns['RUNS_PER_WORKLOAD']} runs, got {len(runs)}")
    for run in runs:
        tag = f"{run['workload']} r{run['run']}"
        if not run["valid"]:
            problems.append(f"{tag}: invalid ({run['invalid_requests'][:3]}, {run['sanity']})")
        if run["mode"] == "sequential":
            v = run["comparison"]["vllm"]
            if not (v["counters_exact"] and v["requests_matching"] == v["requests"]):
                problems.append(f"{tag}: vllm model not exact against itself: {v}")
            lru = run["comparison"]["lru"]
            if run["workload"] == "W1_long" and lru["counters_exact"]:
                problems.append(f"{tag}: lru matched W1 — the comparison cannot discriminate")
        elif run["comparison"]["lru_zero_preemptions_holds"] is None:
            problems.append(f"{tag}: concurrent claims not evaluated")
    if not ns.get("SWEEP") or not all(s["valid"] for s in ns["SWEEP"]):
        problems.append("decode sweep missing or invalid")
    if not r["zip_written"] and r["error"] is None:
        problems.append("phase3_results.zip was not written")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vllm-src", type=pathlib.Path, help="a real vllm package dir, for the audit")
    ap.add_argument("--keep", action="store_true", help="keep each scenario's working directory")
    ap.add_argument("--only", choices=list("ABCD"), action="append")
    args = ap.parse_args()
    todo = args.only or list("ABCD")
    failed = False

    if "A" in todo:
        r = run_scenario("A", args.vllm_src, {}, None, args.keep)
        problems = ([r["error"]] if r["error"] else []) + _check_a(r)
        print(f"=== A  the real setup                 {r['seconds']:>6.1f}s  "
              f"{'PASS' if not problems else 'FAIL'}")
        for line in r["stdout"].splitlines():
            if "==>" in line or "claimed" in line or "decode step" in line or "exact" in line:
                print("    |", line.strip())
        for p in problems:
            print("    ⛔", p)
        if problems:
            print(r["stdout"][-3000:])
        failed |= bool(problems)

    if "B" in todo:
        r = run_scenario("B", args.vllm_src, {"MOCK_NO_RESET": "1"}, 1, args.keep)
        problems = ([r["error"]] if r["error"] else []) + _check_a(r)
        if r["ns"].get("RESET_MODE") != "restart":
            problems.append(f"RESET_MODE is {r['ns'].get('RESET_MODE')!r}, expected 'restart'")
        print(f"=== B  no reset endpoint → restarts  {r['seconds']:>6.1f}s  "
              f"{'PASS' if not problems else 'FAIL'}")
        for p in problems:
            print("    ⛔", p)
        failed |= bool(problems)

    if "C" in todo:
        r = run_scenario("C", args.vllm_src, {"MOCK_IGNORE_OVERRIDE": "1"}, 1, True)
        ok = r["error"] is not None and "tokens of KV cache" in r["error"]
        print(f"=== C  override ignored → must stop  {r['seconds']:>6.1f}s  "
              f"{'PASS' if ok else 'FAIL'}   ({r['error']})")
        failed |= not ok

    if "D" in todo:
        # A stale mock already owns the port. Ours must die with "address in use" and the notebook
        # must refuse to carry on against the stale one.
        stale = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    f"import sys; sys.path[:0]=[{str(REPO / 'scripts')!r},{str(REPO / 'src')!r}]; "
                    "from phase3_dryrun import mock_main; "
                    "mock_main(['serve','m','--num-gpu-blocks-override','512'])"
                ),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(2)
            r = run_scenario("D", args.vllm_src, {}, 1, True)
            ok = r["error"] is not None and "already serving on port" in r["error"]
            print(f"=== D  stale server on the port       {r['seconds']:>6.1f}s  "
                  f"{'PASS' if ok else 'FAIL'}   ({r['error']})")
            failed |= not ok
        finally:
            stale.terminate()
            stale.wait(timeout=10)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
