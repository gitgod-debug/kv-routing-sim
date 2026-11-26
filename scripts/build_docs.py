#!/usr/bin/env python3
"""Render docs/ as HTML pages for the site.

    python scripts/build_docs.py --out _site/docs

`build_site.py` runs this. Each Markdown page becomes one HTML page with the same links: a link to
another page of docs/ goes to its HTML page, a figure is copied beside it, and a link to any other
file of the repository goes to that file on GitHub. A link that leads nowhere stops the build.
Needs `markdown-it-py` (the `docs` extra).
"""

from __future__ import annotations

import argparse
import html
import posixpath
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
GITHUB = "https://github.com/gitgod-debug/kv-routing-sim"
START = "docs/00-START-HERE.md"  # this one becomes index.html
IMAGES = (".gif", ".png", ".svg", ".jpg", ".webp")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · kv-routing-sim</title>
<link rel="stylesheet" href="{root}../style.css">
<link rel="stylesheet" href="{root}docs.css">
</head>
<body>
<header class="top"><a href="{root}../index.html">kv-routing-sim</a> <span>docs</span>
<a href="{github}">GitHub</a></header>
<div class="layout">
<nav aria-label="Documentation"><details open><summary>Pages</summary>
<ul>
{nav}
</ul></details></nav>
<main>
{body}
<footer>Rendered from <a href="{github}/blob/main/{source}">{source}</a>.</footer>
</main>
</div>
</body>
</html>
"""

CSS = """.top { display: flex; gap: 12px; align-items: baseline; padding: 14px 16px;
       border-bottom: 1px solid var(--line); }
.top span { color: var(--muted); }
.top a:last-child { margin-left: auto; }
.layout { display: grid; grid-template-columns: 240px minmax(0, 1fr); max-width: 1180px;
          margin: 0 auto; }
.layout nav { padding: 24px 8px 24px 16px; font-size: 0.93rem; }
.layout nav summary { color: var(--muted); cursor: pointer; }
.layout nav ul { list-style: none; margin: 8px 0; padding: 0; }
.layout nav li { margin: 4px 0; }
.layout nav [aria-current] { font-weight: 600; color: var(--fg); text-decoration: none; }
.layout main { max-width: none; min-width: 0; overflow-wrap: anywhere; }
.layout main h2 { border-top: 1px solid var(--line); padding-top: 20px; }
.layout main h3 { margin: 28px 0 4px; font-size: 1.03rem; }
blockquote { margin: 12px 0; padding: 2px 14px; border-left: 4px solid var(--line);
             color: var(--muted); }
.layout table { display: block; overflow-x: auto; }
img { max-width: 100%; }
@media (max-width: 820px) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  .layout nav { order: 2; padding: 0 16px 32px; }  /* below the page on a phone */
}
"""


class BrokenLink(RuntimeError):
    """A link in docs/ that leads to no file of the repository."""


@dataclass(frozen=True)
class Page:
    source: str  # path in the repository, e.g. docs/upstream/README.md
    target: str  # path under the output folder, e.g. upstream/index.html
    title: str


def slug(text: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def pages(files: list[str]) -> list[Page]:
    """Every Markdown page under docs/, in reading order."""
    out = []
    for source in sorted(f for f in files if f.startswith("docs/") and f.endswith(".md")):
        inner = source.removeprefix("docs/")
        if source == START:
            target = "index.html"
        elif inner.endswith("README.md"):
            target = posixpath.join(posixpath.dirname(inner), "index.html")
        else:
            target = inner.removesuffix(".md") + ".html"
        first = next((line for line in (REPO / source).read_text(encoding="utf-8").splitlines()
                      if line.startswith("# ")), f"# {inner}")
        out.append(Page(source, target, re.sub(r"[`*]", "", first[2:]).strip()))
    return out


def link(href: str, page: Page, by_source: dict[str, Page], files: set[str]) -> str:
    """Where a link of `page` goes on the site."""
    if re.match(r"^([a-z][a-z0-9+.-]*:|//|#)", href):
        return href
    path, _, anchor = href.partition("#")
    repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(page.source), path))
    tail = f"#{anchor}" if anchor else ""
    here = posixpath.dirname(page.target)
    if repo_path in by_source:
        return posixpath.relpath(by_source[repo_path].target, here or ".") + tail
    if repo_path.startswith("docs/") and repo_path.lower().endswith(IMAGES) and repo_path in files:
        return posixpath.relpath(repo_path.removeprefix("docs/"), here or ".")
    if repo_path in files:
        return f"{GITHUB}/blob/main/{repo_path}{tail}"
    if any(f.startswith(repo_path.rstrip("/") + "/") for f in files):
        return f"{GITHUB}/tree/main/{repo_path.rstrip('/')}{tail}"
    raise BrokenLink(f"{page.source}: {href}")


def render(page: Page, all_pages: list[Page], files: set[str]) -> str:
    """One page as HTML."""
    by_source = {p.source: p for p in all_pages}
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    tokens = md.parse((REPO / page.source).read_text(encoding="utf-8"))
    used: dict[str, int] = {}
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            base = slug(_text(tokens[i + 1]))
            n = used.get(base, 0)
            used[base] = n + 1
            token.attrSet("id", base if n == 0 else f"{base}-{n}")
        for child in token.children or []:
            for name in ("href", "src"):
                if name in child.attrs:
                    child.attrSet(name, link(str(child.attrs[name]), page, by_source, files))
    here = posixpath.dirname(page.target)
    nav = "\n".join(
        f'<li><a href="{posixpath.relpath(p.target, here or ".")}"'
        + (' aria-current="page"' if p == page else "") + f">{html.escape(p.title)}</a></li>"
        for p in all_pages)
    return PAGE.format(title=html.escape(page.title), root="../" * page.target.count("/"),
                       github=GITHUB, nav=nav, source=page.source,
                       body=md.renderer.render(tokens, md.options, {}))


def _text(inline: Token) -> str:
    return "".join(c.content for c in inline.children or [] if c.type in ("text", "code_inline"))


def tracked_files() -> list[str]:
    """The repository's files, from git where there is one, else from the folder."""
    out = subprocess.run(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=False)
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout.splitlines()
    return [p.relative_to(REPO).as_posix() for p in REPO.rglob("*")
            if p.is_file() and not {".git", ".venv", "__pycache__"} & set(p.parts)]


def build(out: Path) -> list[Page]:
    files = tracked_files()
    known, found = set(files), pages(files)
    for page in found:
        target = out / page.target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(page, found, known), encoding="utf-8")
    for name in files:
        if name.startswith("docs/") and name.lower().endswith(IMAGES):
            copy = out / name.removeprefix("docs/")
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, copy)
    (out / "docs.css").write_text(CSS, encoding="utf-8")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "_site" / "docs")
    out = parser.parse_args().out.resolve()
    print(f"rendered {len(build(out))} pages into {out}")


if __name__ == "__main__":
    main()
