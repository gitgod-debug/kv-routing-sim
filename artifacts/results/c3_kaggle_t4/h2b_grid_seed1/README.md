# Not the registered H2b-C2 run — the same grid on another calibration

`h2b_c2.json` and `report.md` here are the output of `scripts/run_h2b_c2.py` with its registered
config unchanged **but for the profile**: instead of C2's fit, the profile `kvsim calibrate` wrote
on a real T4 in its SECOND run (C3 part 3) — added post hoc, after the first comparison reversed H2b's verdict.

⚠️ `report.md`'s title and its "registered prediction" line are that script's fixed text. The
registered test of H2b on C2's profile is [`../../h2b_c2/`](../../h2b_c2/), and it was FALSIFIED.
With the engine of release 0.2.0 the rules fail here too. With the engine of 0.1.2 they held
here, and those files are kept in [`../../v0.1.2/`](../../v0.1.2/README.md). The three
calibrations side by side are in
[`../posthoc_three_calibrations.md`](../posthoc_three_calibrations.md).
