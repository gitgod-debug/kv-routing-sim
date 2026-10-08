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
- **Two published routing policies, as ports, asked for by name** (`--policies
  lmetric,smetric-local`; `all` stays the seven above). `lmetric` — LMetric (Zhang et al.,
  OSDI '26), from its authors' router: the replica with the smallest (prefill tokens queued there
  + this request's own after its cache hit) × (running and queued requests + 1); no weight to
  tune. `smetric-local` — SMetric's session rule (Wang et al., arXiv 2607.08565, its Figure 27)
  **without its global cache tier**: a follow-up stays on the replica that holds its history
  unless that replica would miss the first-token target; everything else is placed by a load
  score. Both read what each replica's cache holds from the engine, as the routers they come from
  do. They are this project's ports: a result from them is not a result about the authors'
  systems, and each file lists where it departs from its source.
- **The cache model** follows vLLM 0.30.0's block pool; plain LRU is kept as an ablation
  (`cache_model="lru"`). Checked against a live vLLM on a Tesla T4: exact on two sequential workloads
  under eviction pressure, +0.3% on a concurrent one.
- **Trace adapters:** `vllm-jsonl`, `openai-log`, `mooncake` (the FAST'25 traces: 512-token
  block hashes, sessions inferred) and `bailian` (Alibaba's public Qwen-Bailian usage traces:
  16-token block hashes, and real sessions from each request's parent). Prompts enter as block
  hashes only; no text is ever read.
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
  latency. **`kvsim replay` sizes on both halves by default:** beside the cost model's first-token
  limit it applies p95 ≤ 100 ms per output token (`--tpot-slo-ms`; `0` sizes on the first token
  alone), unless the cost model sets its own `tpot_p95_ms` under `slo:`. The library and the
  shipped cost model are unchanged, so every registered result is what it was (ADR-013).
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
  range widened by one replica each way. Size for the top of the range, plus one — `kvsim replay`
  says this itself beside its recommendation.
- **The README's tables are first-token counts; `kvsim replay`'s default target has both
  halves**, and needs about twice the GPUs. The per-token model was checked on a live T4 (C4):
  p95 time per output token within 7% on three workloads around a 100 ms limit — on one replica.
  At the knee of the load curve it is 1.5–1.7× too slow (T1): decode steps of very large batches
  are 12–15% faster than the fitted line, and the knee multiplies that.
- First-token latency on light traffic came out 1.23–1.30× the model's number in C4 (the model
  too fast), inside the 2× it was tested for.
- `kvsim calibrate` times prompts of up to 2,048 tokens and decode contexts of up to 1,024 by
  default. Longer prompts are an extrapolation unless you measure there (`--prefill-lengths`,
  `--prefill-cached`, `--decode-contexts`).
- `--format html` is not built; `md` and `json` are.
- **Real traces on the validated timing model:** Mooncake was re-run (session-sticky and
  balanced prefix-affinity within one replica, on two calibrations). AgentX could not be: at the
  scale it was registered with, 28% of its prompts are longer than the engine's 2,048-token limit,
  and round-robin never holds the target. Its earlier result is from the first timing model.
- **The two published policies are ports, and nobody else has reviewed them.** `lmetric` was
  written from its authors' router and `smetric-local` from the paper's pseudocode.
  `smetric-local` has no global cache tier — a session that moves recomputes its history — and
  the paper says its lead depends on one. It also leaves out the paper's filter on token speed.
  The balanced-affinity policy here is a hand-weighted sum, the design LMetric argues against.
  On the one public trace they were run on (R3, first hour of the Qwen-Bailian coder trace,
  scaled 1/16), neither needed more replicas than session-sticky under a first-token target, and
  LMetric needed fewer on one calibration of four; with a limit on token speed as well, both
  needed two to three fewer at the highest load — a reported number, not a tested rule.
- **Generated sessions all have the same number of turns.** Session-sticky routing is known to
  unbalance the load when sessions differ widely in size (SMetric); traffic from `kvsim generate`
  cannot show that. Use a real trace to check it. The Qwen-Bailian hour run in R3 did not show it
  either (the busiest replica got at most 1.14× its share): its sessions are short.
- NVIDIA's AISimulate is a larger simulator for the same question, with timing data for many
  data-centre GPUs. See the README's scope section for how the two differ.
