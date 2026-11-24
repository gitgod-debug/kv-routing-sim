"""Tests for the link rewriting of the release build (scripts/build_dist.py).

PyPI shows the README but cannot follow a relative link. The build itself is not run here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
HOME = "https://github.com/example/project"
RAW = "https://raw.githubusercontent.com/example/project"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dist = _load("build_dist")


def _rewrite(text: str) -> str:
    return dist.absolute_links(text, HOME, "v1.2.3")


def test_a_relative_page_link_points_at_the_tag_on_github():
    assert _rewrite("[log](docs/09.md)") == f"[log]({HOME}/blob/v1.2.3/docs/09.md)"
    assert _rewrite("[x](docs/a.md#part-2)") == f"[x]({HOME}/blob/v1.2.3/docs/a.md#part-2)"
    assert _rewrite("[folder](docs/upstream/)") == f"[folder]({HOME}/tree/v1.2.3/docs/upstream/)"


def test_an_image_is_loaded_as_a_raw_file():
    assert _rewrite("![demo](docs/run.gif)") == f"![demo]({RAW}/v1.2.3/docs/run.gif)"
    html = '<source srcset="docs/figures/dark.svg"><img src="docs/figures/light.svg" width="100%">'
    assert _rewrite(html) == (f'<source srcset="{RAW}/v1.2.3/docs/figures/dark.svg">'
                              f'<img src="{RAW}/v1.2.3/docs/figures/light.svg" width="100%">')


def test_absolute_links_and_places_on_the_same_page_are_left_alone():
    for text in ("[demo](https://example.org/run.html)", "[finding](#the-finding)",
                 "[mail](mailto:someone@example.org)", '<a href="https://example.org">x</a>'):
        assert _rewrite(text) == text


def test_no_relative_link_is_left_in_this_readme():
    meta = dist.project()
    text = dist.absolute_links((REPO / "README.md").read_text(encoding="utf-8"),
                               meta["urls"]["Homepage"], f"v{meta['version']}")
    targets = (re.findall(r"\]\(([^)\s]+)\)", text)
               + re.findall(r'(?:src|srcset|href)="([^"]+)"', text))
    assert len(targets) > 50
    assert [t for t in targets if not re.match(r"^(https?:|mailto:|#)", t)] == []
    raw = meta["urls"]["Homepage"].replace("github.com", "raw.githubusercontent.com")
    assert f"{raw}/v{meta['version']}/docs/figures/" in text
