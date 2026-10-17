"""Trace ingestion: adapters read each input format into one versioned record type (ADR-004).

`schema` defines `NormalizedRequest`, `adapters` holds one reader per format, `generator` makes
synthetic agentic traffic and `hashing` turns prompts into chained block hashes. Prompts are hashed
at the adapter boundary and never enter the simulator, which bounds memory and keeps user content
out of every result.
"""
