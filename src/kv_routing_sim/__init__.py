"""kv-routing-sim: choose a KV-cache routing policy and a replica count for an LLM serving fleet.

A trace-driven simulator of a multi-replica vLLM fleet with a recommender on top. The cache model
follows vLLM 0.30.0's block pool and was checked against a live engine.

    traces/    trace adapters and the agentic traffic generator; prompts become block hashes
    cache/     the vLLM block-pool model and a plain LRU model kept as an ablation
    policies/  routing policies, one class per file, behind a registry
    sim/       replicas, the event loop and the timing profile
    eval/      sweeps, cost, the Pareto frontier, bootstrap intervals and the recommendation
    calib/     `kvsim calibrate`: fit a timing profile to a running engine
    service    the replay workflow shared by the command line and the browser page
    cli        the `kvsim` command line

See docs/05-codebase-map.md for the module contracts.
"""

__version__ = "0.1.0"  # keep in step with pyproject.toml
