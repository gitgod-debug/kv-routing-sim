# H6, ranking half — does the order of the policies survive a change of hardware?

Phase 4's design (2 traces × 6 loads × 7 policies × replicas 1–8 × 5 seeds) under 5 regimes · SLO p95 TTFT ≤ 2,000 ms · generated `2026-09-27T19:01:41+00:00` at `ff2ba87`.

⚠️ **Only `t4` is a measured profile.** Every other regime rescales its timings and describes no real GPU, so its milliseconds and rupees are not predictions. Only the ORDER of the policies is under test.

## Verdict

| regime | prefill × | decode × | flips | min τ-b | points with a τ-b | verdict |
|---|---:|---:|---:|---:|---:|---|
| faster_2x | 0.5 | 0.5 | 0 | 0.516 | 8 of 12 | FALSIFIED |
| faster_4x | 0.25 | 0.25 | 0 | 0.653 | 8 of 12 | FALSIFIED |
| cheap_prefill | 0.5 | 1.0 | 0 | 0.653 | 10 of 12 | FALSIFIED |
| dear_prefill | 2.0 | 1.0 | 1 | 0.510 | 10 of 12 | FALSIFIED |

**H6, ranking half: DOES NOT HOLD.** It holds only if every rescaled regime has zero flips and every τ-b is at least 0.8.

*Flip:* the baseline's recommendation is not among the regime's lowest-replica policies, so following it on that hardware would cost a replica. *τ-b:* Kendall's rank agreement between the replica counts each policy needs, baseline vs regime (— where either side is all ties).

Reported, not tested: the literal reading (the regime's own pick names a different policy) and a kinder one (the two best sets share no policy).

| regime | strict flips | lenient flips |
|---|---:|---:|
| faster_2x | 4 | 0 |
| faster_4x | 7 | 0 |
| cheap_prefill | 2 | 0 |
| dear_prefill | 2 | 0 |

## Replicas each policy needs to hold the SLO

### t4 · low_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 3 | 4 | 5 | 7 | 8 |
| least-loaded | 1 | 3 | 4 | 5 | 7 | 8 |
| power-of-two | 1 | 3 | 4 | 5 | 7 | 8 |
| consistent-hash-prefix | 1 | — | — | — | — | — |
| radix-affinity | 1 | — | — | — | — | — |
| radix-affinity-balanced | 1 | 2 | 3 | 3 | 4 | 5 |
| session-sticky | 1 | 2 | 3 | 3 | 4 | 5 |

— = not met at 8 replicas.

### t4 · high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 3 | 4 | 5 | 7 | — |
| least-loaded | 1 | 3 | 4 | 5 | 7 | 8 |
| power-of-two | 1 | 3 | 4 | 5 | 7 | 8 |
| consistent-hash-prefix | 1 | — | — | — | — | — |
| radix-affinity | 1 | — | — | — | — | — |
| radix-affinity-balanced | 1 | — | — | — | — | — |
| session-sticky | 1 | 2 | 3 | 3 | 4 | 5 |

— = not met at 8 replicas.

### faster_2x · low_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 1 | 2 | 3 | 4 | 5 |
| least-loaded | 1 | 1 | 2 | 3 | 4 | 5 |
| power-of-two | 1 | 1 | 2 | 3 | 4 | 5 |
| consistent-hash-prefix | 1 | 1 | — | — | — | — |
| radix-affinity | 1 | 1 | — | — | — | — |
| radix-affinity-balanced | 1 | 1 | 2 | 2 | 3 | 3 |
| session-sticky | 1 | 1 | 2 | 2 | 3 | 3 |

— = not met at 8 replicas.

### faster_2x · high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 1 | 2 | 3 | 4 | 5 |
| least-loaded | 1 | 1 | 2 | 3 | 4 | 5 |
| power-of-two | 1 | 1 | 2 | 3 | 4 | 5 |
| consistent-hash-prefix | 1 | 1 | — | — | — | — |
| radix-affinity | 1 | 1 | — | — | — | — |
| radix-affinity-balanced | 1 | 1 | 2 | 2 | — | — |
| session-sticky | 1 | 1 | 2 | 2 | 3 | 3 |

— = not met at 8 replicas.

### faster_4x · low_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 1 | 2 | 2 | 2 | 3 |
| least-loaded | 1 | 1 | 2 | 2 | 2 | 3 |
| power-of-two | 1 | 1 | 2 | 2 | 2 | 3 |
| consistent-hash-prefix | 1 | 1 | — | — | — | — |
| radix-affinity | 1 | 1 | — | — | — | — |
| radix-affinity-balanced | 1 | 1 | 2 | 2 | 2 | 3 |
| session-sticky | 1 | 1 | 2 | 2 | 2 | 3 |

— = not met at 8 replicas.

### faster_4x · high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 1 | 2 | 2 | 2 | 3 |
| least-loaded | 1 | 1 | 2 | 2 | 2 | 3 |
| power-of-two | 1 | 1 | 2 | 2 | 2 | 3 |
| consistent-hash-prefix | 1 | 1 | — | — | — | — |
| radix-affinity | 1 | 1 | — | — | — | — |
| radix-affinity-balanced | 1 | 1 | 2 | 2 | — | — |
| session-sticky | 1 | 1 | 2 | 2 | 2 | 3 |

— = not met at 8 replicas.

### cheap_prefill · low_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 2 | 3 | 4 | 5 | 6 |
| least-loaded | 1 | 2 | 3 | 4 | 5 | 6 |
| power-of-two | 1 | 2 | 3 | 4 | 5 | 6 |
| consistent-hash-prefix | 1 | — | — | — | — | — |
| radix-affinity | 1 | — | — | — | — | — |
| radix-affinity-balanced | 1 | 2 | 2 | 3 | 4 | 4 |
| session-sticky | 1 | 2 | 2 | 3 | 4 | 4 |

— = not met at 8 replicas.

### cheap_prefill · high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 1 | 2 | 3 | 4 | 5 | 6 |
| least-loaded | 1 | 2 | 3 | 4 | 5 | 6 |
| power-of-two | 1 | 2 | 3 | 4 | 5 | 6 |
| consistent-hash-prefix | 1 | — | — | — | — | — |
| radix-affinity | 1 | — | — | — | — | — |
| radix-affinity-balanced | 1 | 2 | — | — | — | — |
| session-sticky | 1 | 2 | 2 | 3 | 4 | 4 |

— = not met at 8 replicas.

### dear_prefill · low_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 2 | 4 | 7 | — | — | — |
| least-loaded | 2 | — | — | — | — | — |
| power-of-two | 3 | 7 | — | — | — | — |
| consistent-hash-prefix | — | — | — | — | — | — |
| radix-affinity | — | — | — | — | — | — |
| radix-affinity-balanced | 2 | 3 | 5 | 6 | 6 | — |
| session-sticky | 2 | 3 | 4 | 5 | 6 | 7 |

— = not met at 8 replicas.

### dear_prefill · high_sharing

| policy | 1/s | 2/s | 3/s | 4/s | 6/s | 8/s |
|---|---|---|---|---|---|---|
| round-robin | 2 | 4 | 7 | — | — | — |
| least-loaded | 3 | — | — | — | — | — |
| power-of-two | 4 | 8 | — | — | — | — |
| consistent-hash-prefix | — | — | — | — | — | — |
| radix-affinity | — | — | — | — | — | — |
| radix-affinity-balanced | — | — | — | — | — | — |
| session-sticky | 2 | 3 | 5 | 5 | 6 | 7 |

— = not met at 8 replicas.

## Every point

| regime | trace @ load | our pick | best there | τ-b | flip |
|---|---|---|---|---:|:---:|
| faster_2x | low_sharing @ 1/s | round-robin | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_2x | low_sharing @ 2/s | radix-affinity-balanced | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_2x | low_sharing @ 3/s | radix-affinity-balanced | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.791 | no |
| faster_2x | low_sharing @ 4/s | session-sticky | radix-affinity-balanced, session-sticky | 1.000 | no |
| faster_2x | low_sharing @ 6/s | session-sticky | radix-affinity-balanced, session-sticky | 1.000 | no |
| faster_2x | low_sharing @ 8/s | session-sticky | radix-affinity-balanced, session-sticky | 1.000 | no |
| faster_2x | high_sharing @ 1/s | round-robin | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_2x | high_sharing @ 2/s | session-sticky | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_2x | high_sharing @ 3/s | session-sticky | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.653 | no |
| faster_2x | high_sharing @ 4/s | session-sticky | radix-affinity-balanced, session-sticky | 0.516 | no |
| faster_2x | high_sharing @ 6/s | session-sticky | session-sticky | 1.000 | no |
| faster_2x | high_sharing @ 8/s | session-sticky | session-sticky | 0.828 | no |
| faster_4x | low_sharing @ 1/s | round-robin | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_4x | low_sharing @ 2/s | radix-affinity-balanced | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_4x | low_sharing @ 3/s | radix-affinity-balanced | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.791 | no |
| faster_4x | low_sharing @ 4/s | session-sticky | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.791 | no |
| faster_4x | low_sharing @ 6/s | session-sticky | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.791 | no |
| faster_4x | low_sharing @ 8/s | session-sticky | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.791 | no |
| faster_4x | high_sharing @ 1/s | round-robin | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_4x | high_sharing @ 2/s | session-sticky | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| faster_4x | high_sharing @ 3/s | session-sticky | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.653 | no |
| faster_4x | high_sharing @ 4/s | session-sticky | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.653 | no |
| faster_4x | high_sharing @ 6/s | session-sticky | least-loaded, power-of-two, round-robin, session-sticky | 0.894 | no |
| faster_4x | high_sharing @ 8/s | session-sticky | least-loaded, power-of-two, round-robin, session-sticky | 0.694 | no |
| cheap_prefill | low_sharing @ 1/s | round-robin | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| cheap_prefill | low_sharing @ 2/s | radix-affinity-balanced | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.791 | no |
| cheap_prefill | low_sharing @ 3/s | radix-affinity-balanced | radix-affinity-balanced, session-sticky | 1.000 | no |
| cheap_prefill | low_sharing @ 4/s | session-sticky | radix-affinity-balanced, session-sticky | 1.000 | no |
| cheap_prefill | low_sharing @ 6/s | session-sticky | radix-affinity-balanced, session-sticky | 1.000 | no |
| cheap_prefill | low_sharing @ 8/s | session-sticky | radix-affinity-balanced, session-sticky | 1.000 | no |
| cheap_prefill | high_sharing @ 1/s | round-robin | consistent-hash-prefix, least-loaded, power-of-two, radix-affinity, radix-affinity-balanced, round-robin, session-sticky | — | no |
| cheap_prefill | high_sharing @ 2/s | session-sticky | least-loaded, power-of-two, radix-affinity-balanced, round-robin, session-sticky | 0.653 | no |
| cheap_prefill | high_sharing @ 3/s | session-sticky | session-sticky | 1.000 | no |
| cheap_prefill | high_sharing @ 4/s | session-sticky | session-sticky | 1.000 | no |
| cheap_prefill | high_sharing @ 6/s | session-sticky | session-sticky | 1.000 | no |
| cheap_prefill | high_sharing @ 8/s | session-sticky | session-sticky | 0.828 | no |
| dear_prefill | low_sharing @ 1/s | round-robin | least-loaded, radix-affinity-balanced, round-robin, session-sticky | — | no |
| dear_prefill | low_sharing @ 2/s | radix-affinity-balanced | radix-affinity-balanced, session-sticky | 0.849 | no |
| dear_prefill | low_sharing @ 3/s | radix-affinity-balanced | session-sticky | 0.775 | ⛔ yes |
| dear_prefill | low_sharing @ 4/s | session-sticky | session-sticky | 0.754 | no |
| dear_prefill | low_sharing @ 6/s | session-sticky | radix-affinity-balanced, session-sticky | 0.791 | no |
| dear_prefill | low_sharing @ 8/s | session-sticky | session-sticky | 0.510 | no |
| dear_prefill | high_sharing @ 1/s | round-robin | round-robin, session-sticky | — | no |
| dear_prefill | high_sharing @ 2/s | session-sticky | session-sticky | 0.800 | no |
| dear_prefill | high_sharing @ 3/s | session-sticky | session-sticky | 0.701 | no |
| dear_prefill | high_sharing @ 4/s | session-sticky | session-sticky | 0.632 | no |
| dear_prefill | high_sharing @ 6/s | session-sticky | session-sticky | 0.632 | no |
| dear_prefill | high_sharing @ 8/s | session-sticky | session-sticky | 0.655 | no |
