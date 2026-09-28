# Contributing

Thank you for looking. This is a small research tool with a few strict habits; they are what make
its numbers worth anything, so a pull request is judged against them first.

## Set up

```bash
git clone <this repo> && cd kv-routing-sim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # Python 3.11+; the core needs only PyYAML
pytest -q && ruff check .        # both must pass; CI runs them on 3.11, 3.12 and 3.13
```

No GPU is needed for anything except calibrating against a live engine.

## The habits

1. **Every number is reproducible by a script.** Anything in `artifacts/results/` comes from a
   script in `scripts/` with pinned seeds, and records the commit it ran at. If you change a result,
   commit the script that made it.
2. **Experiments are registered before they run.** A new hypothesis gets its design in `configs/`
   and a dated entry in [`docs/09-hypotheses.md`](docs/09-hypotheses.md), committed *before* the run;
   the result is a new dated entry, falsified or not. Earlier entries are never edited.
3. **No prompt text, ever.** Traces enter as chains of block hashes. An adapter hashes at its
   boundary and never writes text anywhere.
4. **No uncited constants.** Every cost constant lives in `configs/cost_model.yaml` with a source and
   a date.
5. **Fail loudly, at startup.** A bad config or an unknown name stops before any work (ADR-004).
6. **Ties are broken by the injected `rng`** — never by `min()` over a list, which quietly prefers
   replica 0 and manufactures cache locality ([`policies/base.py`](src/kv_routing_sim/policies/base.py)).

## Common changes

**A new routing policy** is one file. Subclass `RoutingPolicy` in
[`src/kv_routing_sim/policies/`](src/kv_routing_sim/policies/), implement `choose(state, request,
rng)`, add it to `REGISTRY` in `registry.py`. The parametrised tests in `tests/test_policies.py`
pick it up; add one test for what makes it different. A policy sees load only (`FleetState`), never
the replicas' caches — a real router does not either.

**A new trace adapter** is one file in
[`src/kv_routing_sim/traces/adapters/`](src/kv_routing_sim/traces/adapters/), registered in that
package's `REGISTRY`. It yields `NormalizedRequest`s, counts every skipped line with a reason in
`TraceStats`, and gets tests in `tests/test_traces.py` with a small inline sample — never a real
trace committed to the repo.

**A new profile** comes from `kvsim calibrate` against your engine. Commit the JSON to
`configs/profiles/` with its `_source` block intact. Only measured profiles are shipped in the
package, and the shipped copy must stay byte-identical (`tests/test_packaging.py`).

## Pull requests

- One change per pull request, with a test that fails without it.
- `pytest -q` and `ruff check .` pass.
- If a committed result moves, say which and why in the description.

Good first issues are listed in [`docs/good-first-issues.md`](docs/good-first-issues.md).
