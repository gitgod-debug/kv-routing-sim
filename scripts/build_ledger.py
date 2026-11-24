#!/usr/bin/env python3
"""Build the README's evidence ledger: each number on the first screen, and where it comes from.

    python scripts/build_ledger.py            # print the table
    python scripts/build_ledger.py --write    # replace the table in README.md
    python scripts/build_ledger.py --check    # exit 1 if README.md does not hold this table

Every row names a claim, the number as the README shows it, the results file that holds it, what
made that file, and the last commit that changed the file. The number is read from the file each
time and compared with the text in the row, so a row cannot say something its file does not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
README = REPO / "README.md"
RESULTS = "artifacts/results"
START, END = "<!-- ledger:start -->", "<!-- ledger:end -->"


@dataclass(frozen=True)
class Row:
    claim: str
    shown: str  # the number as the README's first screen shows it
    file: str
    made_by: str
    read: Callable[[Any], str]  # the same number, formatted from the file's content


def _max_factor(runs: list[dict], key: str) -> float:
    return max(max(r[key].values()) if isinstance(r[key], dict) else r[key] for r in runs)


def _session(doc: dict, target: str, policy: str) -> dict:
    cell = doc["headline_cell"]
    return next(r for r in doc["rows"]
                if (r["target"], r["policy"], r["prompts"], r["load"])
                == (target, policy, cell["prompts"], cell["load"]))


def _cost(doc: dict, target: str, policy: str) -> str:
    r = _session(doc, target, policy)
    return (f"{r['replicas']} T4s, {r['re_prefilled_tokens_per_session']:.0f} tokens, "
            f"₹{r['inr_per_session']:.3f}")


def _rule_outside_its_cell(doc: dict) -> str:
    """S1c's ratio over every cell that has a row under both targets."""
    both: dict[tuple, dict[str, dict]] = {}
    for r in doc["rows"]:
        both.setdefault((r["prompts"], r["load"], r["policy"]), {})[r["target"]] = r
    pairs = [v for v in both.values() if len(v) == 2]
    off = sum(
        1 for v in pairs
        if abs((v["two_part"]["inr_per_session"] / v["first_token"]["inr_per_session"])
               / (v["two_part"]["replicas"] / v["first_token"]["replicas"]) - 1.0) > 0.10)
    return f"{off} of {len(pairs)} cells"


ROWS = [
    Row("Cache model exact on two workloads, three runs each", "3 of 3 runs, on both",
        f"{RESULTS}/phase3_kaggle_t4/phase3.json", "notebooks/phase3_calibration.ipynb",
        lambda d: "3 of 3 runs, on both" if all(
            d["verdict"][w]["vllm"]["exact_runs"] == d["verdict"][w]["runs"] == 3
            for w in ("W1_long", "W2_short")) else "not exact"),
    Row("A third, concurrent workload (H3)", "+0.3%",
        f"{RESULTS}/phase3_kaggle_t4/fitted_profile.json", "scripts/phase3_analyze.py",
        lambda d: f"{d['calibration_error']:+.1%}"),
    Row("What a naive cache model misses (H4)", "−11.9 points",
        f"{RESULTS}/phase3_kaggle_t4/phase3.json", "notebooks/phase3_calibration.ipynb",
        lambda d: f"−{abs(d['verdict']['W1_long']['lru']['hit_rate_error_pts'][0]):.1f} points"),
    Row("Hit rates on two live replicas (Phase 6)", "within 1.4%",
        f"{RESULTS}/phase6_kaggle_t4/analysis.json", "scripts/phase6_analyze.py",
        lambda d: f"within {max(d['verdict_run1']['s1_max'], d['verdict_run2']['s1_max']):.1%}"),
    Row("Latency on those two replicas, first timing model", "5–222× off",
        f"{RESULTS}/phase6_kaggle_t4/analysis.json", "scripts/phase6_analyze.py",
        lambda d: "{:.0f}–{:.0f}× off".format(*d["p95_ratio_measured_over_predicted"])),
    Row("The calibration command's cache check (C1)", "48 of 48 requests",
        f"{RESULTS}/c1_kaggle_t4/c1.json", "notebooks/c1_calibrate_t4.ipynb",
        lambda d: f"{d['cached_tokens_equal']} of {d['cached_tokens_compared']} requests"),
    Row("Timing: three predictions (C2)", "1.92×, 1.00×, within 1.41×",
        f"{RESULTS}/c2_kaggle_t4/analysis.json", "scripts/c2_analyze.py",
        lambda d: (f"{d['c2a']['ratio']:.2f}×, {d['c2b']['median_ratio']:.2f}×, "
                   f"within {_max_factor(d['c2c']['runs'], 'factor'):.2f}×")),
    Row("p95 time to first token on traffic the fit never saw", "within 1.4×",
        f"{RESULTS}/c2_kaggle_t4/analysis.json", "scripts/c2_analyze.py",
        lambda d: f"within {_max_factor(d['c2c']['runs'], 'factor'):.1f}×"),
    Row("p95 time per output token around the limit (C4)", "within 7%",
        f"{RESULTS}/c4_kaggle_t4/analysis.json", "scripts/c4_analyze.py",
        lambda d: f"within {max(r['factor'] for r in d['traffic']['runs']) - 1:.0%}"),
    Row("Nine rules for per-token timing (C4)", "7 of 9 held",
        f"{RESULTS}/c4_kaggle_t4/analysis.json", "scripts/c4_analyze.py",
        lambda d: "{} of {} held".format(
            sum(bool(d[k]["holds"]) for k in d["predicted_outcomes"]),
            len(d["predicted_outcomes"]))),
    Row("One calibration's own spread covers the next one's GPU count (C4, rule E)",
        "82.4% of cases",
        f"{RESULTS}/c4_kaggle_t4/analysis.json", "scripts/c4_analyze.py",
        lambda d: f"{d['e']['coverage']:.1%} of cases"),
    Row("Per-token latency at the knee of the load curve (T1)", "1.54× and 1.71×",
        f"{RESULTS}/t1_itl/t1.json", "scripts/t1_itl_check.py",
        lambda d: "{:.2f}× and {:.2f}×".format(
            *sorted(r["A"]["factor"] for r in d["runs"] if r["A"]["factor"] > 1.5))),
    Row("Per-token latency off the knee (T1)", "within 1.07×",
        f"{RESULTS}/t1_itl/t1.json", "scripts/t1_itl_check.py",
        lambda d: "within {:.2f}×".format(
            max(r["A"]["factor"] for r in d["runs"] if r["A"]["factor"] <= 1.5))),
    Row("GPU counts under the two-part target (T2)", "1.93–2.00× the first-token counts",
        f"{RESULTS}/t2_two_part_slo/analysis.json", "scripts/t2_analyze.py",
        lambda d: "{:.2f}–{:.2f}× the first-token counts".format(
            min(c["median_ratio"] for c in d["t2c"]["by_calibration"].values()),
            max(c["median_ratio"] for c in d["t2c"]["by_calibration"].values()))),
    Row("A GPU count is good to ± one", "102 of 144 counts equal on three calibrations, 138 "
        "within one",
        f"{RESULTS}/c3_kaggle_t4/posthoc_three_calibrations.json", "scripts/c3_posthoc_grid.py",
        lambda d: (f"{d['cells_identical_on_all']} of {d['points'] * len(d['policies'])} counts "
                   f"equal on three calibrations, {d['cells_within_one']} within one")),
    Row("Ranking of policies across hardware (H6)", "τ-b down to 0.51",
        f"{RESULTS}/h6_ranking/h6.json", "scripts/run_h6.py",
        lambda d: f"τ-b down to {min(v['min_tau_b'] for v in d['verdicts'].values()):.2f}"),
    Row("Two published policies on a production trace (R3)", "2 of 4 held",
        f"{RESULTS}/r3_bailian/analysis.json", "scripts/r3_analyze.py",
        lambda d: "{} of 4 held".format(sum(bool(d[k]["holds"])
                                            for k in ("r3a", "r3b", "r3c", "r3d")))),
    Row("New tokens in a session", "519 of 6,800",
        f"{RESULTS}/session_cost/session_cost.json", "scripts/session_cost.py",
        lambda d: "{:.0f} of {:,.0f}".format(
            _session(d, "first_token", "session-sticky")["new_tokens_per_session"],
            _session(d, "first_token", "session-sticky")["prompt_tokens_per_session"])),
    Row("Session-sticky, first-token target", "5 T4s, 27 tokens, ₹0.047",
        f"{RESULTS}/session_cost/session_cost.json", "scripts/session_cost.py",
        lambda d: _cost(d, "first_token", "session-sticky")),
    Row("Round-robin, first-token target", "7 T4s, 669 tokens, ₹0.063",
        f"{RESULTS}/session_cost/session_cost.json", "scripts/session_cost.py",
        lambda d: _cost(d, "first_token", "round-robin")),
    Row("Session-sticky, two-part target", "10 T4s, 61 tokens, ₹0.091",
        f"{RESULTS}/session_cost/session_cost.json", "scripts/session_cost.py",
        lambda d: _cost(d, "two_part", "session-sticky")),
    Row("Round-robin, two-part target", "14 T4s, 834 tokens, ₹0.126",
        f"{RESULTS}/session_cost/session_cost.json", "scripts/session_cost.py",
        lambda d: _cost(d, "two_part", "round-robin")),
    Row("S1's rupee rule outside its registered cell", "more than 10% off in 4 of 86 cells",
        f"{RESULTS}/session_cost/session_cost.json", "scripts/session_cost.py",
        lambda d: f"more than 10% off in {_rule_outside_its_cell(d)}"),
    Row("Memory for the block hashes of a 100× trace", "537 MiB for 80,000 requests",
        f"{RESULTS}/scale/scale.json", "scripts/measure_scale.py",
        lambda d: "{:.0f} MiB for {:,} requests".format(*next(
            (c["peak_rss_mib_after_trace"], c["requests"]) for c in d["cases"]
            if c["case"].startswith("100×") and c["policy"] == "round-robin"))),
    Row("Engine, block size and cache of the timing profile", "vLLM 0.30.0, 16 tokens, 1,537 "
        "blocks",
        "configs/profiles/t4-qwen3-4b-vllm0.30.0-ctx.json", "scripts/c2_analyze.py",
        lambda d: (f"vLLM {d['engine_version']}, {d['block_size']} tokens, "
                   f"{d['kv_cache_blocks']:,} blocks")),
]


def last_commit(path: str) -> str:
    """Short id of the last commit that changed `path`, or '' where git cannot say."""
    out = subprocess.run(
        ["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "log", "-1", "--format=%h",
         "--", path], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def verify(row: Row) -> None:
    """Raise if the row's file is missing, was made by something else, or holds another number."""
    for needed in (row.file, row.made_by):
        if not (REPO / needed).exists():
            raise SystemExit(f"error: {needed} does not exist ({row.claim})")
    doc = json.loads((REPO / row.file).read_text())
    header = doc.get("script") if isinstance(doc, dict) else None
    if header and header != row.made_by:
        raise SystemExit(f"error: {row.file} says it was made by {header}, the ledger says "
                         f"{row.made_by}")
    got = row.read(doc)
    if got != row.shown:
        raise SystemExit(f"error: {row.claim}: the README shows {row.shown!r}, "
                         f"{row.file} holds {got!r}")


def table(commit_of: Callable[[str], str] = last_commit) -> str:
    lines = ["| Claim | Number | Results file | Made by | Commit |", "|---|---|---|---|---|"]
    for row in ROWS:
        verify(row)
        name = pathlib.Path(row.file).name
        lines.append(f"| {row.claim} | {row.shown} | [`{name}`]({row.file}) | "
                     f"`{row.made_by}` | `{commit_of(row.file)}` |")
    return "\n".join(lines)


def in_readme() -> str:
    text = README.read_text(encoding="utf-8")
    return text[text.index(START) + len(START):text.index(END)].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="replace the table in README.md")
    ap.add_argument("--check", action="store_true", help="fail if README.md differs")
    args = ap.parse_args()
    built = table()
    if args.check:
        shown = in_readme()
        commits_known = all(last_commit(row.file) for row in ROWS)
        if not commits_known:
            # No git history here (a copy, or a shallow clone): compare everything but the commits.
            def drop(text: str) -> list[str]:
                return [line.rsplit(" | ", 1)[0] for line in text.splitlines()]
            shown, built = drop(shown), drop(built)
        if shown != built:
            print("README.md does not hold the ledger this script builds; run with --write")
            return 1
        print(f"README.md holds the ledger: {len(ROWS)} rows, each number read from its file"
              + ("" if commits_known else "; commits not checked, there is no git history here"))
        return 0
    if args.write:
        text = README.read_text(encoding="utf-8")
        head, rest = text.split(START, 1)
        README.write_text(f"{head}{START}\n{built}\n{END}{rest.split(END, 1)[1]}",
                          encoding="utf-8")
        print(f"wrote {len(ROWS)} rows into README.md")
        return 0
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
