# L1: the whole AgentX trace at its recorded size

Registered before it ran (docs/09-hypotheses.md). Made by `scripts/run_l1.py`; the
numbers are in `l1.json` beside this file. Cache accounting only: each request finishes
before the next arrives, so nothing here is a latency, a GPU count or a cost.

**Trace.** 739 coding-agent sessions (747 with sub-agents), 53,967 requests to each session's main model, 7,782,643,452 prompt tokens in 121,577,073 blocks of 64 tokens, 24,014,952 output tokens; the longest prompt has 894,371. 48 sessions are replayed at a time.
**Overlap left out.** Recorded calls take 143.3 of 3,003.7 session-hours, so on average 2.3 of 48 sessions are in a call at any time.
**Ceiling.** 96.77% of the prompt tokens were sent before by the same session, so a cache that never forgot would serve them. The rest is new text.

| Cache per replica | Policy | Hit rate | Computed again, share of prompt tokens (95% interval over sessions) | Minus round-robin (paired) | Evictions |
|---:|---|---:|---:|---:|---:|
| 1 Mi tokens | round-robin | 50.1% | 46.64% (44.85% to 48.41%) |  | 60,862,981 |
| 1 Mi tokens | session-sticky | 95.5% | 1.32% (1.14% to 1.51%) | -45.3 points (-47.0 to -43.7) | 5,747,913 |
| 1 Mi tokens | radix-affinity | 95.5% | 1.30% (1.11% to 1.51%) | -45.3 points (-47.0 to -43.7) | 5,725,817 |
| 2 Mi tokens | round-robin | 64.1% | 32.64% (31.12% to 34.19%) |  | 43,709,355 |
| 2 Mi tokens | session-sticky | 96.6% | 0.18% (0.14% to 0.23%) | -32.5 points (-34.0 to -31.0) | 4,237,674 |
| 2 Mi tokens | radix-affinity | 96.6% | 0.18% (0.13% to 0.23%) | -32.5 points (-34.0 to -31.0) | 4,232,336 |
| 4 Mi tokens | round-robin | 73.1% | 23.65% (22.43% to 24.88%) |  | 32,514,345 |
| 4 Mi tokens | session-sticky | 96.7% | 0.02% (0.02% to 0.03%) | -23.6 points (-24.9 to -22.4) | 3,782,486 |
| 4 Mi tokens | radix-affinity | 96.7% | 0.03% (0.02% to 0.04%) | -23.6 points (-24.8 to -22.4) | 3,789,356 |

## The four predictions

- **L1a held.** With one replica that never evicts, 53,967 of 53,967 requests' cached tokens equal the count made from the raw file.
- **L1b held.** The longest run of the grid took 5.4 minutes on one core (limit 10).
- **L1c held.** Round-robin computes again 35.42× at 1 Mi, 178.8× at 2 Mi, 993.16× at 4 Mi what session-sticky does (at least 4× predicted).
- **L1d held.** Prefix-affinity and session-sticky are 0.018 points apart at 1 Mi, 0.004 points apart at 2 Mi, 0.006 points apart at 4 Mi (at most 2 predicted).

Reading the trace took 81 s and 1,303 MiB; the replica that never evicts took 183 s and 3,573 MiB. 34 runs in 44 minutes on 4 workers.
