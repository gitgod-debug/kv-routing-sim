# Results of release 0.1.2

This folder keeps the stored results of release 0.1.2 that release 0.2.0 changed. Each file is
byte for byte what 0.1.2 wrote, at the path it had under `artifacts/results/`.

Release 0.2.0 fixed a defect in the simulator's engine. When a step finished all the work a
replica had, the next request could start a step before that step had ended. A real engine runs
one step at a time. In the same case the router saw the replica as idle while it was still
computing, and a request was counted as finished when its last step started.

Every result that the fix can change was derived again with the fixed engine, as registered in
[`configs/rederive_0_2_0.yaml`](../../../configs/rederive_0_2_0.yaml). The new files are at their
usual paths under `artifacts/results/`. The README says what changed.

To compute these numbers again, check out tag `v0.1.2` and run the same script, or install
release 0.1.2 (`pip install kv-routing-sim==0.1.2`). In the current code the old engine is still
there for single runs: `run(..., engine_version="0.1.2")`, or `with use_engine("0.1.2"):` in the
same process. The grid scripts start worker processes. From Python 3.14 on, and on macOS, those
do not inherit it. So use the tag for a grid.

A file that 0.2.0 did not change is not copied here. It is the same in both releases.
