"""Calibration against a live vLLM engine.

`calibrate` probes a running engine, fits a timing profile to it and checks the cache model against
the engine's prefix-cache counters, from which the hit rate is derived (there is no hit-rate gauge,
ADR-008); `fit` holds the least-squares fit. Nothing here imports vLLM: the engine is reached over
HTTP with the standard library (ADR-009).
"""
