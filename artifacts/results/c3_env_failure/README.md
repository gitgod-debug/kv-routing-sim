# C3's first attempt — it did not start — and the probe that checked the repair

Nothing in this folder is a measurement of the engine. It is the record of why C3's first attempt
on Kaggle (2026-10-05) stopped before `kvsim calibrate` ran, and of the check that the repair works
on Kaggle's image. The story and what was amended: `docs/09-hypotheses.md`, entry of 2026-10-05.

| File | What it is |
|---|---|
| `attempt1_error.txt` | The output of the cell that stopped, as it was pasted back from the Kaggle session. No zip exists for this attempt: the stop came before any cell that writes one. |
| `probe_preflight.json` | What the pre-flight cell recorded on Kaggle's image, in a probe notebook that starts no engine (`scripts/c3_probe.py`, run by `scripts/kaggle_run.py`): the environment, what it found broken, what it did, and `pip freeze` afterwards. Verbatim. |
| `probe_stdout.txt` | The probe notebook's printed output: the package versions as Kaggle ships them, after `pip install vllm==0.30.0`, and after the pre-flight. pip's download progress bars (~440 KB) are dropped; nothing else is. |
| `probe_kaggle_run.json` | How the probe run ended, as `scripts/kaggle_run.py` recorded it. |
