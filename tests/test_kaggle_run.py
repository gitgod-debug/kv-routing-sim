"""Offline tests for scripts/kaggle_run.py, which starts GPU runs on a Kaggle account.

A push is always private, a refused push stops, waiting ends on a final status, and no Kaggle token
is ever in the repository.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "scripts" / "kaggle_run.py")
kr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kr)


class _Api:
    """Stands in for the Kaggle client: records the push, answers statuses from a list."""

    CONFIG_NAME_USER = "username"

    def __init__(self, error=None, statuses=()):
        self.error, self.statuses, self.seen = error, list(statuses), {}

    def get_config_value(self, key):
        return "someone"

    def kernels_push(self, folder, timeout, acc):
        folder = pathlib.Path(folder)
        self.seen = {"meta": json.loads((folder / "kernel-metadata.json").read_text()),
                     "files": sorted(p.name for p in folder.iterdir()), "accelerator": acc}
        return types.SimpleNamespace(error=self.error, version_number=1, url="https://example")

    def kernels_status(self, ref):
        s = self.statuses.pop(0)
        if isinstance(s, Exception):
            raise s
        return types.SimpleNamespace(status=s, failure_message="why" if "ERROR" in s else "")


def test_a_push_is_private_on_the_t4_machine_with_internet(tmp_path):
    nb = tmp_path / "x.ipynb"
    nb.write_text("{}")
    api = _Api()
    assert kr.push(api, nb, "kvsim-test", kr.T4) == "someone/kvsim-test"
    meta = api.seen["meta"]
    assert meta["is_private"] is True  # there is no option to make it public
    assert meta["enable_internet"] is True and meta["enable_gpu"] is True
    assert meta["id"] == "someone/kvsim-test" and meta["code_file"] == "x.ipynb"
    assert api.seen["accelerator"] == "NvidiaTeslaT4"
    assert api.seen["files"] == ["kernel-metadata.json", "x.ipynb"]  # nothing else is uploaded


def test_a_cpu_run_asks_for_no_gpu(tmp_path):
    nb = tmp_path / "x.ipynb"
    nb.write_text("{}")
    api = _Api()
    kr.push(api, nb, "kvsim-test", None)
    assert api.seen["meta"]["enable_gpu"] is False and api.seen["accelerator"] is None


def test_a_refused_push_stops_with_kaggles_reason(tmp_path):
    nb = tmp_path / "x.ipynb"
    nb.write_text("{}")
    with pytest.raises(SystemExit, match="weekly GPU quota"):
        kr.push(_Api(error="weekly GPU quota reached"), nb, "kvsim-test", kr.T4)


def test_waiting_ends_on_a_final_status_and_survives_a_dropped_call():
    """One failed status call must not lose track of a run that is spending GPU time."""
    api = _Api(statuses=["KernelWorkerStatus.QUEUED", ConnectionError("dropped"),
                         "KernelWorkerStatus.RUNNING", "KernelWorkerStatus.ERROR"])
    assert kr.wait(api, "someone/x", poll_s=0, timeout_s=60) == ("KernelWorkerStatus.ERROR", "why")
    api = _Api(statuses=["KernelWorkerStatus.RUNNING", "KernelWorkerStatus.COMPLETE"])
    assert kr.wait(api, "someone/x", poll_s=0, timeout_s=60) == ("KernelWorkerStatus.COMPLETE", "")


def test_after_a_push_the_previous_runs_status_does_not_end_the_wait():
    """Kaggle may report the previous run as COMPLETE for a moment after a push; taking that as
    this run's end would fetch the old results while the new run is still going."""
    stale = ["KernelWorkerStatus.COMPLETE", "KernelWorkerStatus.COMPLETE"]
    api = _Api(statuses=[*stale, "KernelWorkerStatus.QUEUED", "KernelWorkerStatus.RUNNING",
                         "KernelWorkerStatus.ERROR"])
    assert kr.wait(api, "someone/x", 0, 60, just_pushed=True) == ("KernelWorkerStatus.ERROR", "why")
    assert api.statuses == []  # every answer was needed: it did not stop at the stale ones
    # Without a push (--fetch-only) the first final status is this run's.
    api = _Api(statuses=["KernelWorkerStatus.COMPLETE"])
    assert kr.wait(api, "someone/x", 0, 60) == ("KernelWorkerStatus.COMPLETE", "")
    # A run too short to be seen running still ends the wait, after the settling time.
    api = _Api(statuses=["KernelWorkerStatus.COMPLETE"] * 2)
    assert kr.wait(api, "someone/x", 0, 60, just_pushed=True, settle_s=-1)[0].endswith("COMPLETE")


def test_waiting_gives_up_without_claiming_the_run_ended():
    api = _Api(statuses=["KernelWorkerStatus.RUNNING"] * 3)
    status, _ = kr.wait(api, "someone/x", poll_s=0, timeout_s=-1)
    assert "gave up waiting" in status and "may still go on" in status


def test_the_readable_log_keeps_what_was_printed_and_drops_only_progress_bars():
    rows = [{"stream_name": "stdout", "time": 1.0, "data": "part 1 saved\n"},
            {"stream_name": "stderr", "time": 1.1, "data": "a warning\n"},
            {"stream_name": "stdout", "time": 1.2, "data": "   ━━━━━━━━ 1.8/1.8 MB eta 0:00:00\n"},
            {"stream_name": "stdout", "time": 1.3, "data": "C3a  exit code 0\n"}]
    assert kr.readable_log(json.dumps(rows)) == "part 1 saved\nC3a  exit code 0\n"
    assert kr.readable_log("plain text, not JSON") == "plain text, not JSON"


def test_the_runs_own_log_is_read_not_a_log_the_notebook_wrote(tmp_path):
    """The engine log that C3's notebook writes into its results folder is not taken for the
    run's own log, though it is the first file ending in .log."""
    rows = [{"stream_name": "stdout", "time": 1.0, "data": "what the notebook printed\n"}]

    class Api(_Api):
        def kernels_output(self, ref, path, force=False, quiet=True):
            out = pathlib.Path(path)
            (out / "results").mkdir()
            (out / "results" / "server_0.log").write_text("the ENGINE's log\n")
            (out / "x.log").write_text(json.dumps(rows))
            return [str(out / "results" / "server_0.log"), str(out / "x.log")], ""

    files = kr.fetch(Api(), "someone/x", tmp_path)
    assert (tmp_path / "notebook_stdout.txt").read_text() == "what the notebook printed\n"
    assert files[-1].endswith("notebook_stdout.txt")


def test_no_kaggle_token_is_in_the_repository():
    """A Kaggle token belongs in ~/.kaggle, never in a tracked file, and the runner neither reads
    nor prints one."""
    hits = subprocess.run(["git", "grep", "-l", "-I", "-E", "KGAT_[0-9a-fA-F]{16,}"], cwd=REPO,
                          capture_output=True, text=True, check=False).stdout.split()
    assert hits == []
    src = (REPO / "scripts" / "kaggle_run.py").read_text()
    assert "os.environ" not in src and "KAGGLE_API_TOKEN" not in src.split('"""')[2]
