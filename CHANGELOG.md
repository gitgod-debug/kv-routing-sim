# Changelog

Versions follow [semantic versioning](https://semver.org/). Before 1.0, the command-line interface and
the file formats may still change between minor versions. Each change will be listed here.

## 0.1.0 — unreleased

The first public version. It is not on PyPI yet.

### Added
- **An in-browser runner** (`site/run.html`). It takes generated traffic or a serving log that you
  choose, the measured T4 profile or one from `kvsim calibrate`, and your own latency target and
  GPU price. The simulator runs in the page through Pyodide, and the file is never uploaded. The
  page prints the `kvsim` commands that reproduce its answer.
- **The replay workflow as a library call**: `kv_routing_sim.service.replay()` returns the report
  that `kvsim replay --format json` writes, with an optional progress callback.
- The options `kvsim replay --ttft-slo-ms` and `--gpu-price-usd`, and
  `kvsim generate --system-prompts`.
- **`kvsim replay`** takes a serving trace and gives a routing policy, a replica count and ₹ per
  1,000 requests. Ties on the replica count are reported, not hidden. Seven policies: round-robin,
  least-loaded, power-of-two, consistent-hash-prefix, radix-affinity, radix-affinity-balanced (with
  a load weight) and session-sticky.
- **Two published routing policies, as ports, asked for by name** (`--policies
  lmetric,smetric-local`; `all` stays the seven above). `lmetric` is LMetric (Zhang et al.,
  OSDI '26), taken from its authors' router. It picks the replica with the smallest (prefill tokens
  queued there + this request's own after its cache hit) × (running and queued requests + 1). It
  has no weight to tune. `smetric-local` is SMetric's session rule (Wang et al., arXiv
  2607.08565, its Figure 27) without its global cache tier. A follow-up stays on the replica that
  holds its history, unless that replica would miss the first-token target. Everything else is
  placed by a load score. Both read from the engine what each replica's cache holds, as the routers
  they come from do. They are this project's ports. A result from them is not a result about the
  authors' systems, and each file lists where it departs from its source.
- **The cache model** follows the block pool of vLLM 0.30.0. A plain LRU model is kept for
  comparison (`cache_model="lru"`). It was checked against a live vLLM on a Tesla T4: exact on two
  sequential workloads under eviction pressure, and +0.3% on a concurrent one.
- **Trace adapters:** `vllm-jsonl`, `openai-log`, `mooncake` (the FAST'25 traces: 512-token block
  hashes, sessions inferred) and `bailian` (Alibaba's public Qwen-Bailian usage traces: 16-token
  block hashes, and real sessions from each request's parent). Prompts enter the simulator only as
  block hashes. `openai-log` is the one adapter that reads message text, on your machine, to make
  those hashes. Its logs have no token ids, so it counts about four characters as one token. This
  can move a block boundary, and so the hit rate. No text is kept or written out.
- **`kvsim calibrate`** fits the context-aware timing model to your own running vLLM. It measures
  decode cells over context length × batch (`--decode-contexts`), and prefill over prompt length,
  cold and after a cached prefix (`--prefill-cached` and `--prefill-new`). It reads the engine's KV
  capacity and checks the cache model against the engine's counters, request by request. It uses
  the standard library only.
- **A context-aware timing model** (C2): prefill and decode cost grow with context length. Both
  terms default to zero, so every earlier profile gives the same results as before.
- **How firm a replica count is.** `kvsim calibrate` measures every timing cell `--reps` times (3)
  and keeps the fit of each repeat on its own in the profile. `kvsim replay` sizes on each fit and
  prints the range of replica counts next to its recommendation (`--no-spread` skips the extra
  sweeps). A profile without repeats says that its spread is unknown.
- **The edge check.** `kvsim replay` runs its recommended configuration once more, with every
  timing coefficient 5% larger. It prints how far the p95 latencies moved: "not at the
  edge", "at the edge" (it stops meeting the SLO, so allow one replica more) or "at the knee" (a
  p95 moves by more than 1.5×, so the count is not to be trusted). This costs one extra run per
  seed, and `--no-edge-check` skips it.
- **Time per output token.** Every run now also measures the time per output token of each
  request: last token − first token, over the tokens after the first. This is MLPerf's TPOT and
  vLLM's `request_time_per_output_token`. `kvsim replay` prints its p95 next to the first-token
  latency. By default, `kvsim replay` sizes on both halves. Next to the cost model's first-token
  limit, it applies p95 ≤ 100 ms per output token (`--tpot-slo-ms`, where `0` sizes on the first
  token alone). The cost model can set its own `tpot_p95_ms` under `slo:` instead. The library and
  the shipped cost model are unchanged, so every registered result is still what it was (ADR-013).
- **`kvsim generate`** (agentic traffic with shared prompts, including many tenants) and
  **`kvsim crossover`** (sharing × load).
- **Shipped data:** the context-aware T4 profile that C2 validated (`--profile
  t4-qwen3-4b-vllm0.30.0-ctx`, the one to use), the first profile, which is context-free and kept
  for the record (`t4-qwen3-4b-vllm0.30.0`), and the cited cost model (`--cost-model default`),
  byte-identical to `configs/`.
- The core install needs only PyYAML.

### Changed
- Reports use words instead of symbols: "yes" and "no" in the "meets SLO" column, and "Warning:"
  and "Note:" labels.

### Fixed
- An unknown policy name, a missing trace file or an out-of-range generator setting now ends with
  an error message instead of a Python traceback.

### Known limits
- Every millisecond, rupee and replica count from the shipped profiles is a **T4 figure**. Run
  `kvsim calibrate` for your own numbers.
- `kvsim calibrate` was proven on a real T4 in both its forms. One is the cache check and the
  capacity reading (C1). The other is the context-aware timing fit, whose profile predicted three
  unseen workloads within 1.41× (C3). Near saturation, one run is not enough. A second run on the
  same engine differed by 3–5% in its coefficients, and that put one p95 4.2× off. Run it more than
  once if your answer sits at the edge of the target.
- How far a replica count can be trusted: in 92 of the 144 cells of the finding's grid, three
  calibrations of the same GPU type all give a count. In those cells, the counts are equal in 58
  and within one in 89. In 8 more cells, the calibrations disagree on whether the target can be
  held at all.
- The "how firm" range is a lower bound, and a low one. On a live T4 (C4), another run of
  `kvsim calibrate` gave a replica count outside a run's own range in 18% of cases. With the range
  widened by one replica each way, it was 5%. Size for the top of the range, plus one.
  `kvsim replay` says this itself next to its recommendation.
- The README's tables are first-token counts. The default target of `kvsim replay` has both
  halves, and it needs about twice the GPUs. The per-token model was checked on a live T4 (C4): p95
  time per output token within 7% on three workloads around a 100 ms limit. This was on one
  replica. At the knee of the load curve, it is 1.5–1.7× too slow (T1). Decode steps of very large
  batches are 12–15% faster than the fitted line, and the knee multiplies that.
- The threshold of the edge check was chosen after looking. On the five held-out GPU runs of T1,
  it flags both runs that the simulator got wrong and none of the three that it got right. Any
  threshold between 1.21× and 5.33× would do the same. These are five runs of three workloads on
  one GPU.
- In C4, first-token latency on light traffic came out at 1.23–1.30× the model's number (the model
  was too fast). This is inside the 2× it was tested for.
- By default, `kvsim calibrate` times prompts of up to 2,048 tokens and decode contexts of up to
  1,024. Longer prompts are an extrapolation unless you measure there (`--prefill-lengths`,
  `--prefill-cached`, `--decode-contexts`).
- `--format html` is not built. `md` and `json` are.
- **Real traces on the validated timing model:** Mooncake was run again (session-sticky and
  balanced prefix-affinity within one replica, on two calibrations). AgentX could not be run again.
  At the scale it was registered with, 28% of its prompts are longer than the engine's 2,048-token
  limit, and round-robin never holds the target. Its earlier result is from the first timing model.
- The two published policies are ports, and nobody else has reviewed them. `lmetric` was written
  from its authors' router, and `smetric-local` from the paper's pseudocode. `smetric-local` has no
  global cache tier, so a session that moves recomputes its history. The paper says that SMetric's
  lead depends on such a tier. `smetric-local` also leaves out the paper's filter on token speed.
  The balanced-affinity policy here is a hand-weighted sum, which is the design LMetric argues
  against. They were run on only one public trace (R3, the first hour of the Qwen-Bailian coder
  trace, scaled 1/16). There, under a first-token target, neither needed more replicas than
  session-sticky, and LMetric needed fewer on one calibration of four. With a limit on token speed
  as well, both needed two to three fewer at the highest load. This is a reported number, not a
  tested rule.
- Generated sessions all have the same number of turns. Session-sticky routing is known to
  unbalance the load when sessions differ widely in size (SMetric). Traffic from `kvsim generate`
  cannot show that, so use a real trace to check it. The Qwen-Bailian hour run in R3 did not show it
  either (the busiest replica got at most 1.14× its share). Its sessions are short.
- NVIDIA's AISim (called AISimulate until November 2026) is a larger simulator for the same
  question, with timing data for many data-centre GPUs. See the README's scope section for how the
  two differ.
