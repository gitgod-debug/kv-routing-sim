#!/usr/bin/env python3
"""Prepare V1: Phase 3's exactness test again, on a later release of the engine.

    python scripts/v034_prepare.py

Reads configs/v034_recheck.yaml, which names Phase 3's config and one difference, the vLLM
version, and writes two files:
  artifacts/results/v034_prereg/predictions.json   (committed before the GPU run)
  notebooks/v034_recheck_t4.ipynb

Both are made by Phase 3's own code (`phase3_prepare.make_predictions` and the notebook builder
in `_phase3_notebook`), so the trace, the predicted numbers, the analysis procedure and the
decision rules are Phase 3's. The script writes nothing unless the trace fingerprint and every
predicted number equal Phase 3's registered ones, and unless the notebook differs from Phase 3's
only in the version, and in the copy of the predictions it carries, which names the version.
"""

from __future__ import annotations

import base64
import json
import lzma
import pathlib
import re
import sys
from datetime import UTC, datetime
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "v034_recheck.yaml"
PREREG = REPO / "artifacts" / "results" / "v034_prereg" / "predictions.json"
NOTEBOOK = REPO / "notebooks" / "v034_recheck_t4.ipynb"
# The embedded data, as the notebook carries it: a run of quoted base64 lines.
_DATA_LINES = re.compile(r'(?:[ \t]*"[A-Za-z0-9+/=]{40,}"\n?)+')


def load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return Phase 3's config and the same config with this test's engine version."""
    spec = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load((REPO / spec["base"]).read_text())
    return base, {**base, "engine": {**base["engine"], "vllm": str(spec["engine"]["vllm"])}}


def notebook_for(p3, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the notebook for `cfg` the way Phase 3 does; return its text and what it embeds."""
    from _phase3_notebook import build_notebook

    trace, prompts, fingerprint, predictions = p3.make_predictions(cfg)
    payload = {
        "trace": [
            {"request_id": r.request_id, "arrival_ts": r.arrival_ts, "session_id": r.session_id,
             "ids": ids}
            for r, ids in trace
        ],
        "decode_sweep_prompts": prompts,
        "predictions": predictions,
    }
    blob = base64.b64encode(lzma.compress(p3.canonical(payload), preset=9)).decode()
    pred_sha = p3.sha256(predictions)
    text = build_notebook(cfg, blob, fingerprint, pred_sha, predictions)
    return text, {"fingerprint": fingerprint, "predictions": predictions, "pred_sha": pred_sha,
                  "sent": {k: payload[k] for k in ("trace", "decode_sweep_prompts")},
                  "n_requests": len(trace)}


def same_but_for_the_version(new: str, old: str, hashes: tuple[str, str],
                             versions: tuple[str, str]) -> bool:
    """True if the notebooks are equal, cell by cell, once three things are masked.

    The three are the engine version, the embedded data and the hash of the predictions. The
    embedded data differs because the predictions inside it name the engine version; `main`
    compares what it decodes to separately.
    """
    def cells(text: str) -> list[tuple[str, str]]:
        out = []
        for cell in json.loads(text)["cells"]:
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            src = _DATA_LINES.sub("<DATA>\n", src)
            for marker, values in (("<HASH>", hashes), ("<VERSION>", versions)):
                for value in values:
                    src = src.replace(value, marker)
            out.append((cell["cell_type"], src))
        return out
    return cells(new) == cells(old)


def main() -> int:
    sys.path.insert(0, str(REPO / "scripts"))
    import phase3_prepare as p3

    base, cfg = load_config()
    old_version, new_version = str(base["engine"]["vllm"]), str(cfg["engine"]["vllm"])
    registered = json.loads(p3.PREREG.read_text())

    text, parts = notebook_for(p3, cfg)
    predictions = parts["predictions"]
    if parts["fingerprint"] != registered["trace_fingerprint_sha256"]:
        raise SystemExit("error: the trace is not the one Phase 3 registered. Nothing written.")
    changed = sorted(k for k in predictions if predictions[k] != registered["predictions"][k])
    if changed != ["engine"] or set(predictions) != set(registered["predictions"]):
        raise SystemExit(f"error: more than the engine differs from Phase 3: {changed}. "
                         "Nothing written.")
    old_text, old_parts = notebook_for(p3, base)
    if parts["sent"] != old_parts["sent"]:
        raise SystemExit("error: the notebook would send other token ids than Phase 3's. "
                         "Nothing written.")
    if not same_but_for_the_version(text, old_text, (parts["pred_sha"], old_parts["pred_sha"]),
                                    (old_version, new_version)):
        raise SystemExit("error: the notebook differs from Phase 3's in more than the version. "
                         "Nothing written.")

    prereg = {
        "what": (f"V1 pre-registered predictions: Phase 3's, for vLLM {new_version}. "
                 "Committed before any GPU run."),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": cfg,
        "base_config": str(yaml.safe_load(CONFIG.read_text())["base"]),
        "only_difference_from_phase3": f"engine.vllm {old_version} -> {new_version}",
        "trace_fingerprint_sha256": parts["fingerprint"],
        "predictions_sha256": parts["pred_sha"],
        "n_requests": parts["n_requests"],
        "sequential_counts_verified_timing_independent":
            registered["sequential_counts_verified_timing_independent"],
        "simulator_assumptions": registered["simulator_assumptions"],
        "analysis_procedure": registered["analysis_procedure"],
        "decision_rules": registered["decision_rules"],
        "predictions": predictions,
    }
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(prereg, indent=1) + "\n")
    NOTEBOOK.write_text(text)
    print(f"wrote {PREREG.relative_to(REPO)} and {NOTEBOOK.relative_to(REPO)}: Phase 3's trace "
          f"and predictions, vLLM {old_version} -> {new_version}, and nothing else")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
