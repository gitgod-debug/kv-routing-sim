"""Tests for the docs pages of the site (scripts/build_docs.py)."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

pytest.importorskip("markdown_it")

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


docs = _load("build_docs")
PAGE = docs.Page("docs/a.md", "a.html", "A")
SUB = docs.Page("docs/upstream/README.md", "upstream/index.html", "Upstream")
BY_SOURCE = {p.source: p for p in (PAGE, SUB)}
FILES = {"docs/a.md", "docs/upstream/README.md", "docs/figures/x.svg", "README.md", "src/k/m.py"}


@pytest.mark.parametrize(("href", "page", "where"), [
    ("upstream/README.md", PAGE, "upstream/index.html"),
    ("../a.md#part-2", SUB, "../a.html#part-2"),
    ("figures/x.svg", PAGE, "figures/x.svg"),
    ("../README.md#the-finding", PAGE,
     "https://github.com/gitgod-debug/kv-routing-sim/blob/main/README.md#the-finding"),
    ("../src/k/", PAGE, "https://github.com/gitgod-debug/kv-routing-sim/tree/main/src/k"),
    ("#here", PAGE, "#here"),
    ("https://example.org", PAGE, "https://example.org"),
])
def test_where_a_link_goes(href, page, where):
    assert docs.link(href, page, BY_SOURCE, FILES) == where


def test_a_link_that_leads_nowhere_stops_the_build():
    with pytest.raises(docs.BrokenLink, match=r"docs/a.md: ../nowhere.md"):
        docs.link("../nowhere.md", PAGE, BY_SOURCE, FILES)


def test_a_heading_gets_the_anchor_github_gives_it():
    assert docs.slug("The finding, in detail") == "the-finding-in-detail"
    assert docs.slug("2a. ⛔ What three passes missed — found 2026-10-05") == \
        "2a--what-three-passes-missed--found-2026-10-05"


def test_every_page_is_rendered_and_every_internal_link_leads_to_a_file(tmp_path):
    found = docs.build(tmp_path)
    assert {p.source for p in found} == {f for f in docs.tracked_files()
                                         if f.startswith("docs/") and f.endswith(".md")}
    for page in found:
        text = (tmp_path / page.target).read_text(encoding="utf-8")
        assert "<table>" in text or page.source.endswith("good-first-issues.md")
        for href in re.findall(r'(?:href|src)="([^"]+)"', text):
            if re.match(r"^(https?:|mailto:|#)", href):
                continue
            path = (tmp_path / page.target).parent / href.split("#")[0]
            outside = path.resolve().relative_to(tmp_path.resolve().parent)
            if outside.parts[0] == tmp_path.name:  # the rest is the site around docs/
                assert path.exists(), f"{page.source}: {href}"
