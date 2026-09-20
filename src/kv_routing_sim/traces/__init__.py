"""Trace ingestion: any input format in, one versioned record type out. 📐 designed, not built.

Contract (ADR-004):
  * `schema.py`     — `NormalizedRequest` + `SCHEMA_VERSION`. Every results file records the
                      version it was produced under, so curves from different weeks are only
                      compared when they are actually comparable.
  * `adapters/`     — one file per format (vLLM JSONL, SGLang, OpenAI-style logs). Each is a
                      GENERATOR; none returns a list. Traces can be large and the memory ceiling
                      is a stated number, not a discovered one.
  * `generator.py`  — synthetic agentic traffic with `prefix_sharing_ratio` as an explicit
                      parameter, because that parameter is the H2 sweep axis.
  * `hashing.py`    — prompt → block hashes, mirroring the engine's hash-chain rule: a block's
                      hash depends on its contents AND its predecessors, so reuse is only possible
                      as a prefix.

Two invariants:
  * The prompt is hashed at the ADAPTER BOUNDARY and never enters the simulator. That is the memory
    bound and also a privacy property — traces are other people's user content, and none of it need
    ever be held in memory or written to an artifact.
  * A malformed line LOGS AND CONTINUES, and the count lands in the results header. A run that
    dropped 12% of its input must say so. A bad *config*, by contrast, fails loudly at startup:
    data is dirty and expected to be, configuration is ours.
"""
