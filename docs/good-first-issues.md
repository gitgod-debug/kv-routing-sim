# Three good first issues — ready to open

Each is self-contained, touches one area, and has a test that decides when it is done. Open them on
GitHub once the repository is public, with the label `good first issue`; the text below is ready to
paste.

---

## 1. A `least-tokens` routing policy — does counting tokens fix least-loaded?

**Why.** Twice now, `least-loaded` needed *more* replicas than plain round-robin: with 64 tenant
prompts at 1 session/s (8 replicas against 5), and with prefill made twice as dear in the H6 test
(see `docs/09-hypotheses.md`, Phase 5 and H6). One untested explanation: it counts *requests*, so a
replica stuck in one long prefill looks lightly loaded and keeps receiving more. A router that
counts outstanding *tokens* would not be fooled that way.

**What to do.**
1. Give `FleetState` (`src/kv_routing_sim/policies/base.py`) a load-only view of tokens: for each
   replica, the prompt tokens not yet prefilled plus the output tokens not yet decoded, over its
   running and queued requests. Fill it where the engine builds `FleetState`
   (`src/kv_routing_sim/sim/engine.py`). No cache information — a real router knows the lengths of
   what it dispatched, not what is resident.
2. Add `least-tokens` in a new file: the replica with the fewest outstanding tokens, ties broken
   from `rng` (`argmin_random`). Register it.
3. Tests: it is picked up by the parametrised policy tests; one new test shows it avoids a replica
   holding one very long prompt that `least-loaded` would pick.

**Done when** the tests pass and `python scripts/run_phase5.py` (or a small config of your own)
shows how many replicas it needs next to `least-loaded` — whatever the answer.

---

## 2. `kvsim replay --format html` — a report you can open in a browser

**Why.** `--format md` and `--format json` exist; the plan promised `html` too. A single
self-contained file is easy to share and needs no renderer.

**What to do.** In `src/kv_routing_sim/cli.py`, add `html` to `--format` and render the same
content as `_markdown()`: the recommendation, the tie line, the warning when uncalibrated, and the
table of every configuration. One file, inline CSS, no JavaScript, no external requests; escape
every string with `html.escape`.

**Done when** a test renders a small replay to HTML and checks that the recommended policy, the
uncalibrated warning (when the profile is not calibrated) and one table row are present — and that
the file contains no `<script>` tag and no `http` link.

---

## 3. Confidence intervals per cell in `kvsim replay`

**Why.** Each configuration in the replay table is an average over seeds, printed without any
uncertainty. The repo already has a bootstrap that resamples **sessions**, not requests
(`src/kv_routing_sim/eval/bootstrap.py`, `bootstrap_ci_grouped`); requests within a session are not
independent, so resampling them gives intervals that are far too narrow (see
`docs/06-methodology.md`).

**What to do.** Add a `--ci` flag to `kvsim replay`. For each (policy, replicas) cell, report a 95%
interval for p95 TTFT by resampling sessions with `bootstrap_ci_grouped`, and show it in the `md`
and `json` outputs. It needs each request's TTFT and session: `run(..., record_per_request=True)`
returns per-request records keyed by `request_id`, and the session ids come from the trace itself.

**Done when** a test shows the interval contains the point estimate, and that it is wider than an
interval computed by resampling requests on a trace where sessions differ strongly — the reason the
grouping exists.
