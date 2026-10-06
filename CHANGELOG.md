# Changelog

Versions follow [semantic versioning](https://semver.org/). Before 1.0, the command-line interface and
the file formats may still change between minor versions; each change will be listed here.

## 0.1.0 — unreleased

The first public version: planned for PyPI before 2026-11-20.

### Added
- **`kvsim replay`** — a serving trace in; a routing policy, a replica count and ₹ per 1,000
  requests out, with ties on the replica count reported rather than hidden. Seven policies:
  round-robin, least-loaded, power-of-two, consistent-hash-prefix, radix-affinity,
  radix-affinity-balanced (with a load weight) and session-sticky.
- **The cache model** follows vLLM 0.30.0's block pool; plain LRU is kept as an ablation
  (`cache_model="lru"`). Checked against a live vLLM on a Tesla T4: exact on two sequential workloads
  under eviction pressure, +0.3% on a concurrent one.
- **Trace adapters:** `vllm-jsonl`, `openai-log`, and `mooncake` (the FAST'25 traces: 512-token
  block hashes, sessions inferred). Prompts enter as block hashes only; no text is ever read.
- **`kvsim calibrate`** — fit the context-aware timing model to your own running vLLM (decode cells
  over context length × batch, `--decode-contexts`; prefill over prompt length, cold and after a
  cached prefix, `--prefill-cached` and `--prefill-new`), read its KV capacity, and check the cache
  model against its counters, request by request. Standard library only.
- **A context-aware timing model** (C2): prefill and decode cost grow with context length. Both
  terms default to zero, so every earlier profile gives the same results as before.
- **How firm is a replica count?** `kvsim calibrate` measures every timing cell `--reps` times (3)
  and keeps the fit of each repeat alone in the profile; `kvsim replay` sizes on each and prints
  the range of replica counts beside its recommendation (`--no-spread` skips the extra sweeps). A
  profile without repeats says its spread is unknown.
- **Time per output token.** Every run now also measures each request's time per output token
  (last token − first token, over the tokens after the first — MLPerf's TPOT, vLLM's
  `request_time_per_output_token`), and `kvsim replay` prints its p95 beside the first-token
  latency. A cost model may add `tpot_p95_ms` under `slo:` to make it the second half of the SLO;
  without that key nothing is enforced on it and every earlier result is unchanged.
- **`kvsim generate`** (agentic traffic with shared prompts, including many tenants) and
  **`kvsim crossover`** (sharing × load).
- **Shipped data:** the context-aware T4 profile C2 validated (`--profile
  t4-qwen3-4b-vllm0.30.0-ctx`, the one to use), the first, context-free one kept for the record
  (`t4-qwen3-4b-vllm0.30.0`), and the cited cost model (`--cost-model default`), byte-identical to
  `configs/`.
- The core install needs only PyYAML.

### Known limits
- Every millisecond, rupee and replica count from the shipped profiles is a **T4 figure**. Run
  `kvsim calibrate` for your own numbers.
- `kvsim calibrate` was proven on a real T4 in both its forms: the cache check and capacity reading
  (C1), and the context-aware timing fit, whose profile predicted three unseen workloads within
  1.41× (C3). **Near saturation one run is not enough:** a second run on the same engine differed by
  3–5% in its coefficients, and that put one p95 4.2× off. Run it more than once if your answer sits
  at the edge of the target.
- A replica count is good to about ± one: three calibrations of the same T4 agree exactly on 71% of
  the counts in the finding's grid and within one on 96%.
- **The "how firm" range is a floor, and a low one.** On a live T4 (C4) another run of
  `kvsim calibrate` gave a replica count outside a run's own range in 18% of cases; 5% with the
  range widened by one replica each way. Size for the top of the range, plus one.
- The default latency target is on the first token only (p95 TTFT). The per-token half
  (`slo.tpot_p95_ms`) is optional and off by default. Its model was checked on a live T4 (C4):
  p95 time per output token within 7% on three workloads around a 100 ms limit — on one replica.
  At the knee of the load curve it is 1.5–1.7× too slow (T1): decode steps of very large batches
  are 12–15% faster than the fitted line, and the knee multiplies that.
- First-token latency on light traffic came out 1.23–1.30× the model's number in C4 (the model
  too fast), inside the 2× it was tested for.
- `kvsim calibrate` times prompts of up to 2,048 tokens and decode contexts of up to 1,024 by
  default. Longer prompts are an extrapolation unless you measure there (`--prefill-lengths`,
  `--prefill-cached`, `--decode-contexts`).
- `--format html` is not built; `md` and `json` are.
- **Two published routing policies are not among the seven:** LMetric (OSDI '26 — new prefill
  tokens × batch size, no weight) and SMetric (arXiv 2607.08565 — stick a session's follow-ups,
  migrate it when its instance would miss the SLO). The balanced-affinity policy here is a
  hand-weighted sum, the design LMetric argues against.
- **Generated sessions all have the same number of turns.** Session-sticky routing is known to
  unbalance the load when sessions differ widely in size (SMetric); traffic from `kvsim generate`
  cannot show that. Use a real trace to check it.
- NVIDIA's AISimulate is a larger simulator for the same question, with timing data for many
  data-centre GPUs. See the README's scope section for how the two differ.
