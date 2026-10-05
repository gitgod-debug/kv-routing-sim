"""The C3 notebook's pre-flight cell (scripts/_c3_notebook.py) — the one cell that is C3's own.

C3's first attempt on Kaggle stopped before any measurement: the image shipped a torchaudio built
for another CUDA than the torch vLLM installs, under the very version number vLLM pins, so pip kept
it and vLLM could not start (docs/09, 2026-10-05). The pre-flight repairs that and stops early when
it cannot. It decides whether a GPU hour is spent, so its logic is tested here, with the package
manager and the interpreter replaced by fakes — nothing in this file installs or removes anything.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("_c3_notebook", REPO / "scripts" / "_c3_notebook.py")
nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb)

BROKEN = {"installed": True, "version": "2.11.0+cu128", "ok": False,
          "error": "RuntimeError: Detected that PyTorch and TorchAudio were compiled with "
                   "different CUDA versions."}
FIXED = {"installed": True, "version": "2.11.0", "ok": True}
GONE = {"installed": False}
# vLLM 0.30.0's own requirement strings, as PyPI serves them.
REQUIRES = ["numpy", "transformers>=5.10.4", "numba==0.65.0", "torch==2.13.0",
            "torchaudio==2.11.0", "torchvision==0.28.0", "torchcodec>=0.14",
            'zentorch==2.13.0.0; extra == "zen"']


def _functions() -> dict:
    """The cell's definitions, executed without its "run it" part — as a notebook would have them
    after the cells above it."""
    ns = {"subprocess": subprocess, "sys": sys, "json": json, "re": re, "pathlib": pathlib,
          "OUT_DIR": "unused", "ENV": {}}
    exec(compile(nb.PREFLIGHT_CELL.split("# ── run it")[0], "preflight_cell", "exec"), ns)  # noqa: S102
    return ns


def _fakes(states: list[dict]):
    """A probe that answers from `states` in turn, and a runner that only records."""
    states, ran = list(states), []

    def probe(mod):
        return states.pop(0)

    def run(cmd, timeout=0):
        ran.append(cmd)
        return 0, ""

    return probe, run, ran


def test_the_pinned_version_comes_from_vllms_own_requirements():
    pinned = _functions()["_pinned"]
    assert pinned("torchaudio", REQUIRES) == "2.11.0"
    assert pinned("torchvision", REQUIRES) == "0.28.0"
    assert pinned("torch", REQUIRES) == "2.13.0"  # and not torchaudio's or torchvision's line
    assert pinned("torchcodec", REQUIRES) is None  # a range is not a pin
    assert pinned("zentorch", REQUIRES) is None  # an optional extra is not installed, so not ours
    assert pinned("torchaudio", []) is None


def test_a_broken_package_gets_the_build_vllm_pins_from_pypi():
    """The Kaggle case: 2.11.0+cu128 satisfies `==2.11.0`, so pip kept it. The repair forces the
    PyPI build back in, touching nothing else — and stops there once it imports."""
    ns = _functions()
    probe, run, ran = _fakes([BROKEN, FIXED])
    now = ns["repair"]("torchaudio", probe=probe, run=run, pinned=lambda m: "2.11.0")
    assert now == FIXED
    assert len(ran) == 1
    args = ran[0][3:]
    assert args[0] == "install" and args[-1] == "torchaudio==2.11.0"
    assert "--no-deps" in args and "--force-reinstall" in args  # torch itself must not move
    assert args[args.index("--index-url") + 1] == "https://pypi.org/simple"
    (step,) = ns["PREFLIGHT"]["repairs"]
    assert step["package"] == "torchaudio" and step["was"] == BROKEN and step["now"] == FIXED


def test_if_the_pinned_build_does_not_import_either_the_package_is_removed():
    ns = _functions()
    probe, run, ran = _fakes([BROKEN, BROKEN, GONE])
    assert ns["repair"]("torchaudio", probe=probe, run=run, pinned=lambda m: "2.11.0") == GONE
    assert [c[3] for c in ran] == ["install", "uninstall"]
    assert ran[1][3:] == ["uninstall", "-y", "-q", "torchaudio"]
    assert len(ns["PREFLIGHT"]["repairs"]) == 2


def test_a_package_vllm_does_not_pin_is_removed_at_once():
    ns = _functions()
    probe, run, ran = _fakes([BROKEN, GONE])
    assert ns["repair"]("torchcodec", probe=probe, run=run, pinned=lambda m: None) == GONE
    assert [c[3] for c in ran] == ["uninstall"]


def test_a_healthy_or_absent_package_is_left_alone():
    ns = _functions()
    for state in (FIXED, GONE):
        probe, run, ran = _fakes([state])
        assert ns["repair"]("torchvision", probe=probe, run=run, pinned=lambda m: "0.28.0") == state
        assert ran == [] and ns["PREFLIGHT"]["repairs"] == []


def test_the_probe_runs_in_a_fresh_interpreter_and_survives_a_module_that_raises(
        tmp_path, monkeypatch):
    """The real probe, on modules made for it: one that imports, one that raises as torchaudio
    did, one that is not there."""
    (tmp_path / "c3_good_mod.py").write_text("x = 1\n")
    (tmp_path / "c3_bad_mod.py").write_text(
        "raise RuntimeError('compiled with different CUDA versions')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # how the fresh interpreter finds them
    ns = _functions()
    assert ns["_probe"]("c3_good_mod") == {"installed": True, "version": None, "ok": True}
    bad = ns["_probe"]("c3_bad_mod")
    assert bad["installed"] and not bad["ok"] and "different CUDA versions" in bad["error"]
    assert ns["_probe"]("c3_no_such_mod") == {"installed": False}


def test_an_early_stop_leaves_a_zip_and_says_nothing_was_measured():
    """Before the client cell runs there is no save(): every stop in the start cell must write the
    diagnostics zip first, or "download the zip" is a lie — as it was on the first attempt."""
    cell = nb._start_cell()
    assert cell.count("diag_zip()") == cell.count("raise Stop(") == 2
    for m in re.finditer(r"raise Stop\(", cell):
        assert cell[:m.start()].rstrip().endswith("diag_zip()")
    assert "Nothing was measured" in cell
    run_part = nb.PREFLIGHT_CELL.split("# ── run it")[1]
    assert run_part.count("raise Stop(") == 3  # no GPU · not a T4 · vLLM cannot start
    assert '"T4" not in' in run_part  # C3 is registered for a T4: no other GPU is measured
    for m in re.finditer(r"raise Stop\(", run_part):
        assert run_part[:m.start()].rstrip().endswith("diag_zip()")
    assert '_run(["vllm", "serve", "--help"]' in run_part  # the check the first attempt failed


def test_the_notebook_says_on_page_one_what_was_amended(tmp_path):
    cfg = json.loads(json.dumps({
        "engine": {"model": "m", "vllm": "0.30.0", "block_size": 16, "num_gpu_blocks": 1537,
                   "max_model_len": 2048, "max_num_seqs": 256, "max_num_batched_tokens": 8192},
        "replicas": 1, "measure_replica": 0,
        "calibrate": {"seeds": [0, 1], "check_sessions": 16, "check_turns": 3,
                      "token_range": [1000, 30000], "max_error": 0.05},
        "rules": {"c3b_max_factor": 2.0, "c3c_max_factor": 2.0},
        "predictions": {"c3a": "holds", "c3b": "holds", "c3c": "holds"}}))
    pred = {"runs": {}, "check": {"0": {"hits": 1, "queries": 2, "requests": 3}}}
    kw = {"wheel_b64": "", "wheel_sha": "0" * 64, "wheel_name": "w.whl", "git_sha": "abc1234"}
    plain = json.loads(nb.build_notebook(cfg, {}, {}, pred, "p" * 64, **kw))
    assert "Amended" not in "".join(plain["cells"][0]["source"])
    amended = json.loads(nb.build_notebook(cfg, {}, {}, pred, "p" * 64, **kw, amendments=[
        {"date_utc": "2026-10-05T10:00:00+00:00", "git_sha": "def5678", "why": "A reason."}]))
    intro = "".join(amended["cells"][0]["source"])
    assert ("**Amended 2026-10-05** (commit `def5678`): A reason. Nothing registered changed"
            in intro)
    kinds = [("".join(c["source"])[:40], c["cell_type"]) for c in amended["cells"]]
    install = next(i for i, (head, _) in enumerate(kinds) if "PINNED" in head)
    assert "Pre-flight" in kinds[install + 1][0] and "Pre-flight" in kinds[install + 2][0]
    assert "@@KVSIM-WHEEL@@" in kinds[install + 3][0]  # the wheel is installed after the check
