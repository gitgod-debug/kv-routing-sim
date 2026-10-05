"""C3's registration must stay reproducible; the notebook must carry exactly what was registered.

The notebook embeds the package's wheel, the held-out traffic's fingerprints and the predictions. If
any of them drifts from artifacts/results/c3_prereg/predictions.json, the GPU run would test
something other than what was registered. A deliberate change means re-running
scripts/c3_prepare.py on a clean tree, and a dated entry in docs/09-hypotheses.md saying why —
before the run, never after.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import itertools
import json
import pathlib
import re

from kv_routing_sim.sim.cost_model import load_profile
from kv_routing_sim.traces.generator import AgenticWorkload

REPO = pathlib.Path(__file__).resolve().parents[1]
PREREG = json.loads((REPO / "artifacts/results/c3_prereg/predictions.json").read_text())
NOTEBOOK = json.loads((REPO / "notebooks/c3_calibrate_t4.ipynb").read_text())
SRC = "".join("".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code")

_spec = importlib.util.spec_from_file_location("c3_prepare", REPO / "scripts" / "c3_prepare.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def _canonical(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def test_the_notebook_carries_the_registered_wheel_traffic_and_predictions():
    wheel = base64.b64decode(re.search(r'b64decode\("""([A-Za-z0-9+/=]+)"""\)', SRC).group(1))
    sha = PREREG["wheel"]["sha256"]
    assert hashlib.sha256(wheel).hexdigest() == sha
    assert f'!= "{sha}"' in SRC  # the notebook checks that hash itself before installing
    assert f'WHEEL_SHA256 = "{sha}"' in SRC  # and writes it into the results it brings back
    assert re.search(r'PREDICTIONS_SHA256 = "([0-9a-f]{64})"', SRC).group(1) == \
        PREREG["predictions_sha256"]
    assert hashlib.sha256(_canonical(PREREG["predictions"])).hexdigest() == \
        PREREG["predictions_sha256"]
    for fp in PREREG["trace_fingerprints_sha256"].values():
        assert fp in SRC
    cal = json.loads(re.search(r'CAL = json.loads\(r"""(.+?)"""\)', SRC).group(1))
    assert cal == PREREG["config"]["calibrate"]
    # Registered from committed code only, and the wheel is that commit's.
    assert PREREG["git_dirty"] is False
    assert PREREG["wheel"]["built_from"] == PREREG["git_sha"]


def test_the_registered_predictions_reproduce_from_the_committed_code():
    cfg = PREREG["config"]
    profile = load_profile(REPO / cfg["profile"])
    for name, spec in PREREG["workloads"].items():
        pairs = list(AgenticWorkload(**spec).iter_with_tokens())
        assert prepare.predict_holdout(pairs, profile) == \
            PREREG["predictions"]["runs"][name]["C2 profile"], name
    for seed in cfg["calibrate"]["seeds"]:
        assert prepare.predict_check(cfg, seed) == PREREG["predictions"]["check"][str(seed)]


def test_c2s_two_workloads_are_the_traces_c2_measured():
    """The config calls them "C2's two, unchanged". That has to be true of the token ids."""
    c2 = json.loads((REPO / "artifacts/results/c2_prereg/predictions.json").read_text())
    for name, fp in c2["trace_fingerprints_sha256"].items():
        assert PREREG["trace_fingerprints_sha256"][name] == fp
    assert set(PREREG["trace_fingerprints_sha256"]) - set(c2["trace_fingerprints_sha256"]) == \
        {"T6_K8"}


def test_every_amendment_is_recorded_and_the_notebook_is_the_last_one():
    """`c3_prepare.py --amend` rebuilds the notebook without touching anything registered — the
    tests above hold either way. Each amendment records why, at which commit, and the notebook's
    hash before and after: so the committed notebook is the registered one, or the last recorded
    amendment of it, and never a silent third thing."""
    amendments = PREREG.get("amendments", [])
    if not amendments:
        return
    text = (REPO / "notebooks/c3_calibrate_t4.ipynb").read_text(encoding="utf-8")
    assert amendments[-1]["notebook_sha256_after"] == hashlib.sha256(text.encode()).hexdigest()
    for earlier, later in itertools.pairwise(amendments):
        assert later["notebook_sha256_before"] == earlier["notebook_sha256_after"]
    intro = "".join(NOTEBOOK["cells"][0]["source"])
    for a in amendments:
        assert "dirty" not in a["git_sha"]  # amended from committed code only
        assert f"(commit `{a['git_sha']}`): {a['why']}" in intro  # and said on page one


def test_the_command_still_measures_the_registered_cells():
    """C3 ran the command with no cell option, so its defaults WERE the registered design.

    C3 has run (2026-10-05), from the wheel its notebook carries — that wheel, not today's source,
    is what the result describes. The cells below are still the command's defaults. One thing has
    changed since, because of what C3 found: the command now measures its decode cells `--reps`
    times, as it always did its prefill cells, and keeps the fit of each repeat. C3 measured them
    once."""
    assert prepare.command_defaults() == PREREG["command_defaults"]
