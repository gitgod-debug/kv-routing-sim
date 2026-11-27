# Contributing

Thank you for looking. This is a small research tool with a few strict habits. These habits are the
reason its numbers can be trusted, so a pull request is checked against them first.

## Set up

```bash
git clone <this repo> && cd kv-routing-sim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # Python 3.11+; the core needs only PyYAML
pytest -q && ruff check .        # both must pass; CI runs them on Python 3.11 to 3.14
```

No GPU is needed for anything except calibrating against a live engine.

## The habits

1. **Reproducible numbers.** Every number can be reproduced by a script. Anything in
   `artifacts/results/` comes from a script in `scripts/` with pinned seeds, and records the commit
   it ran at. If you change a result, commit the script that made it.
2. **Registration before the run.** Experiments are registered before they run. A new hypothesis
   gets its design and its predictions in `configs/`, committed before the run. The result is
   reported, falsified or not. The tree also stays
   unchanged while a registered run is in progress: nothing under `src/`, `scripts/` or `configs/`
   is edited until the run has finished. A results file records whether the tree was clean when
   the run started. A claim that "the edit could not have mattered" is only an argument, and it
   does not replace that record. (Learned on 2026-10-05: T2's grids B and C carry a dirty header
   for this reason. To show that it did not matter, 720 of their cells had to be run again.)
3. **No prompt text, ever.** Traces enter as chains of block hashes. An adapter hashes at its
   boundary and never writes text anywhere.
4. **No uncited constants.** Every cost constant lives in `configs/cost_model.yaml` with a source and
   a date.
5. **Clear errors at startup.** A bad config or an unknown name stops the program before any work
   is done (ADR-004).
6. **Ties are broken by the injected `rng`**, never by `min()` over a list. `min()` quietly prefers
   replica 0, and so it creates artificial cache locality
   ([`policies/base.py`](src/kv_routing_sim/policies/base.py)).

## Common changes

**A new routing policy** is one file. Subclass `RoutingPolicy` in
[`src/kv_routing_sim/policies/`](src/kv_routing_sim/policies/), implement `choose(state, request,
rng)`, add it to `REGISTRY` in `registry.py`. The parametrised tests in `tests/test_policies.py`
pick it up. Add one test for what makes it different. By default, a policy sees only the load
(`FleetState`) and keeps its own record of what it sent where, as a router without cache events
does. A policy that sets `wants_cache_view` is given each replica's cache hits when it decides, as a
router fed by vLLM's KV events sees them. Say which of the two a result used.

**A new trace adapter** is one file in
[`src/kv_routing_sim/traces/adapters/`](src/kv_routing_sim/traces/adapters/), registered in that
package's `REGISTRY`. It yields `NormalizedRequest`s and counts every skipped line, with a reason,
in `TraceStats`. Its tests go in `tests/test_traces.py` and use a small inline sample. Never commit
a real trace to the repository.

**A new profile** comes from `kvsim calibrate` against your engine. Commit the JSON to
`configs/profiles/` with its `_source` block intact. Only measured profiles are shipped in the
package, and the shipped copy must stay byte-identical (`tests/test_packaging.py`).

## Pull requests

- One change per pull request, with a test that fails without it.
- `pytest -q` and `ruff check .` pass.
- If a committed result changes, say which one and why in the description.

Issues that suit a first contribution carry the label `good first issue`.
