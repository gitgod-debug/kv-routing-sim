"""kv-routing-sim — which KV-cache routing policy should you actually run, and on how many replicas?

⚠️ Phase 0: this package is a SCAFFOLD. Each subpackage carries its designed contract as a
docstring and no implementation. That ordering is deliberate — the layout is a set of decisions
(ADR-001…ADR-004), and writing the contracts down before the code exists is what stops them
drifting into whatever the first implementation happened to do.

⛔ No module here may be implemented before the Phase-0 spike passes on a live engine. See
`CHECKPOINTS.md`. The reason is specific rather than procedural: the cache model's block-hashing
rule and the prefill cost function are both meant to MIRROR a real engine, and a mirror built from
documentation is how a competitor ended up shipping a README that cites a metric which does not
exist (`docs/04-prior-art.md`).

Read `docs/05-codebase-map.md` for the module-by-module contract.
"""

__version__ = "0.0.1"
