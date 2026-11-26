"""The measured profiles, keyed by what they were measured on, and how far a setup is from them.

A timing profile holds for one combination of GPU, model, engine version, block size and batch
limit. The nearest measured one is found by counting which of the five differ; nothing is
interpolated between profiles, because an interpolated figure would have no measured error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from . import data

# What stops holding when a field differs, in the order that decides which profile is nearest.
MEANING = {
    "gpu": "timings, replica counts and costs do not carry over",
    "model": "timings and the number of KV blocks do not carry over",
    "engine_version": "timings have not been measured on it, and its cache rules may differ",
    "block_size": "the cache model's block counts, and so hit rates, do not carry over",
    "max_num_seqs": "how many requests run at once, and so latency under load, differs",
}


@dataclass(frozen=True, slots=True)
class Setup:
    """What a profile is measured on. None means unknown, which counts as a difference."""

    gpu: str | None = None
    model: str | None = None
    engine_version: str | None = None
    block_size: int | None = None
    max_num_seqs: int | None = None


@dataclass(frozen=True, slots=True)
class MeasuredProfile:
    name: str
    key: Setup
    kv_cache_blocks: int
    longest_prompt: int
    checked: tuple[str, ...]
    evidence: str


@dataclass(frozen=True, slots=True)
class Difference:
    field: str
    yours: Any
    measured: Any

    @property
    def meaning(self) -> str:
        return MEANING[self.field]


def measured_profiles() -> list[MeasuredProfile]:
    """The profiles shipped with the package, as their index describes them."""
    index = yaml.safe_load(data.profile_index().read_text(encoding="utf-8"))
    return [MeasuredProfile(name=p["name"], key=Setup(**p["key"]),
                            kv_cache_blocks=p["kv_cache_blocks"],
                            longest_prompt=p["longest_prompt"], checked=tuple(p["checked"]),
                            evidence=p["evidence"])
            for p in index["profiles"]]


def differences(yours: Setup, measured: Setup) -> list[Difference]:
    """The fields in which a setup differs from the one a profile was measured on."""
    out = []
    for name in MEANING:
        mine, theirs = getattr(yours, name), getattr(measured, name)
        if mine is None or _normal(mine) != _normal(theirs):
            out.append(Difference(name, mine, theirs))
    return out


def nearest(yours: Setup, profiles: list[MeasuredProfile] | None = None
            ) -> tuple[MeasuredProfile, list[Difference]]:
    """The measured profile with the fewest differences; the GPU, then the model, break ties."""
    profiles = measured_profiles() if profiles is None else profiles
    if not profiles:
        raise ValueError("there is no measured profile to compare with")
    order = list(MEANING)

    def rank(p: MeasuredProfile) -> tuple:
        found = {d.field for d in differences(yours, p.key)}
        return (len(found), [name in found for name in order])

    best = min(profiles, key=rank)
    return best, differences(yours, best.key)


def _normal(value: Any) -> str:
    """Compare without case or a leading 'v', so "v0.30.0" is 0.30.0 and "tesla t4" is Tesla T4."""
    text = str(value).strip().lower()
    return text.removeprefix("v") if text[:2].startswith("v") and text[1:2].isdigit() else text


def longest_prompt_measured(spec: str | Path) -> int | None:
    """The longest prompt whose timing a profile was fitted on, or None if it does not say.

    A shipped profile has it in the index. One written by `kvsim calibrate` has it in its cells.
    """
    path = Path(spec)
    if not path.is_file():
        name = str(spec).removesuffix(".json")
        return next((p.longest_prompt for p in measured_profiles() if p.name == name), None)
    raw = json.loads(path.read_text(encoding="utf-8"))
    source = raw.get("_source") or {} if isinstance(raw, dict) else {}
    seen = [c.get("context", 0) for c in source.get("decode_cells") or []]
    seen += [p.get("prompt_tokens", 0) for p in source.get("prefill_points") or []]
    return max(seen, default=0) or None


def setup_of(**values: Any) -> Setup:
    """A Setup from command-line or engine values, ignoring the ones that were not given."""
    known = {f.name for f in fields(Setup)}
    return Setup(**{k: v for k, v in values.items() if k in known and v not in (None, "")})


def render_list(profiles: list[MeasuredProfile]) -> str:
    """Every measured profile, what it was measured on and what was checked on it."""
    lines = []
    for p in profiles:
        k = p.key
        lines += [
            p.name,
            (f"  measured on   {k.gpu} · {k.model} · vLLM {k.engine_version} · blocks of "
             f"{k.block_size} tokens · max_num_seqs {k.max_num_seqs}"),
            (f"  cache         {p.kv_cache_blocks:,} blocks; prompts up to "
             f"{p.longest_prompt:,} tokens timed"),
            *(f"  checked       {c}" for c in p.checked),
            f"  evidence      {p.evidence}",
            "",
        ]
    lines.append("To see how far your setup is from these: kvsim profiles --gpu ... --model ... "
                 "--engine-version ... --block-size ... --max-num-seqs ...")
    return "\n".join(lines)


def render_distance(yours: Setup, profile: MeasuredProfile, found: list[Difference]) -> str:
    """How a setup compares with its nearest measured profile, field by field."""
    labels = {"gpu": "GPU", "model": "model", "engine_version": "engine version",
              "block_size": "block size", "max_num_seqs": "max_num_seqs"}
    by_field = {d.field: d for d in found}
    lines = [f"Nearest measured profile: {profile.name}"]
    for name in MEANING:
        measured = getattr(profile.key, name)
        if name not in by_field:
            lines.append(f"  {labels[name]:15s} {measured}  same")
        elif by_field[name].yours is None:
            lines.append(f"  {labels[name]:15s} not given; measured on {measured}")
        else:
            lines.append(f"  {labels[name]:15s} {by_field[name].yours}  differs: measured on "
                         f"{measured}")
    if not found:
        lines.append("All five are the same: this profile was measured on your setup. Another "
                     "calibration of it can still differ; see the README on replica counts.")
    else:
        lines.append(f"{len(found)} of {len(MEANING)} differ or are unknown:")
        lines += [f"  {labels[d.field]}: {d.meaning}." for d in found]
        lines.append("Nothing is interpolated between profiles. Measure your own: kvsim calibrate "
                     "--engine <your vLLM URL> --out mine.json")
    return "\n".join(lines)
