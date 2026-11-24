#!/usr/bin/env python3
"""Build the release files for PyPI and check them. Nothing is uploaded.

    python scripts/build_dist.py             # -> dist/kv_routing_sim-<version>.tar.gz and .whl
    python scripts/build_dist.py --no-install-check

PyPI shows the README as the project page, but it cannot follow a relative link or load a relative
image. So the build runs on a copy of the tracked files in which every relative link of the README
points at this version's tag on GitHub. The README in the repository is left as it is.

After building it checks three things:
  1. `twine check` accepts both files;
  2. the source package holds src/, tests/, the README, the licence and pyproject.toml, and
     nothing else;
  3. the wheel installs into an empty environment with PyYAML alone, and `kvsim generate` and
     `kvsim replay` run from it.
Needs the `build` and `twine` packages. Uploading is a separate step, done by hand.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv

REPO = pathlib.Path(__file__).resolve().parents[1]
SDIST_MAY_HOLD = ("src/", "tests/", "README.md", "LICENSE", "pyproject.toml", "PKG-INFO",
                  "setup.cfg")
IMAGE = (".gif", ".png", ".svg", ".jpg", ".jpeg", ".webp")
# A Markdown link or image, and the three HTML attributes the README uses.
_MARKDOWN = re.compile(r"(\]\()([^)\s]+)(\))")
_HTML = re.compile(r'((?:src|srcset|href)=")([^"]+)(")')


def project() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def absolute_links(text: str, homepage: str, tag: str) -> str:
    """Point every relative link at `tag` on GitHub: pages through /blob/, images as raw files."""
    raw = homepage.replace("https://github.com/", "https://raw.githubusercontent.com/")

    def target(link: str) -> str:
        if re.match(r"^([a-z][a-z0-9+.-]*:|#|//)", link):
            return link  # already absolute, or a place on the same page
        path, _, anchor = link.partition("#")
        if path.lower().endswith(IMAGE):
            return f"{raw}/{tag}/{path}"
        kind = "tree" if path.endswith("/") else "blob"
        return f"{homepage}/{kind}/{tag}/{path}" + (f"#{anchor}" if anchor else "")

    for pattern in (_MARKDOWN, _HTML):
        text = pattern.sub(lambda m: m.group(1) + target(m.group(2)) + m.group(3), text)
    return text


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if (REPO / f).is_file()]


def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    done = subprocess.run(argv, capture_output=True, text=True, check=False, **kwargs)
    if done.returncode != 0:
        raise SystemExit(f"error: {' '.join(argv[:4])} … failed:\n{done.stdout}{done.stderr}")
    return done


def build(out: pathlib.Path, tag: str) -> list[pathlib.Path]:
    """Build the source package and the wheel from a copy with an absolute-link README."""
    meta = project()
    with tempfile.TemporaryDirectory(prefix="kvsim-dist-") as scratch:
        source = pathlib.Path(scratch) / "source"
        for name in tracked_files():
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, source / name)
        readme = source / "README.md"
        readme.write_text(absolute_links(readme.read_text(encoding="utf-8"),
                                         meta["urls"]["Homepage"], tag), encoding="utf-8")
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("kv_routing_sim-*"):
            old.unlink()
        run([sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(out),
             str(source)])
    return sorted(out.glob("kv_routing_sim-*"))


def check_sdist(sdist: pathlib.Path) -> int:
    with tarfile.open(sdist) as tar:
        names = [n.split("/", 1)[1] for n in tar.getnames() if "/" in n]
    # A folder is listed once by itself ("src") and then with each file in it ("src/...").
    extra = [n for n in names if n and f"{n}/" not in SDIST_MAY_HOLD
             and not n.startswith(SDIST_MAY_HOLD)]
    if extra:
        raise SystemExit(f"error: the source package holds files it should not: {extra[:10]}")
    return len(names)


def check_install(wheel: pathlib.Path) -> str:
    """Install the wheel into a new environment and run the two commands a new user runs first."""
    with tempfile.TemporaryDirectory(prefix="kvsim-install-") as scratch:
        env_dir = pathlib.Path(scratch) / "env"
        venv.create(env_dir, with_pip=True)
        py = env_dir / "bin" / "python"
        run([str(py), "-m", "pip", "install", "--quiet", str(wheel)])
        installed = run([str(py), "-m", "pip", "list", "--format=freeze"]).stdout.split()
        others = sorted(p.split("==")[0].lower() for p in installed
                        if p.split("==")[0].lower() not in ("pip", "kv-routing-sim", "pyyaml"))
        if others:
            raise SystemExit(f"error: the wheel pulled in more than PyYAML: {others}")
        trace = pathlib.Path(scratch) / "trace.jsonl"
        kvsim = str(env_dir / "bin" / "kvsim")
        run([kvsim, "generate", "--sessions", "12", "--out", str(trace)], cwd=scratch)
        replay = run([kvsim, "replay", str(trace), "--policies", "round-robin,session-sticky",
                      "--replicas", "1-2", "--seeds", "1"], cwd=scratch)
        return replay.stdout.strip().splitlines()[-1] if replay.stdout.strip() else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "dist")
    ap.add_argument("--no-install-check", action="store_true",
                    help="skip installing the wheel into a new environment")
    args = ap.parse_args()
    meta = project()
    tag = f"v{meta['version']}"
    files = build(args.out.resolve(), tag)
    sdist = next(f for f in files if f.name.endswith(".tar.gz"))
    wheel = next(f for f in files if f.suffix == ".whl")
    run([sys.executable, "-m", "twine", "check", "--strict", *map(str, files)])
    print(f"built {sdist.name} ({check_sdist(sdist)} files) and {wheel.name}; twine check passed")
    print(f"the README's relative links point at {meta['urls']['Homepage']}/blob/{tag}/")
    if not args.no_install_check:
        last = check_install(wheel)
        print("the wheel installs with PyYAML alone; `kvsim generate` and `kvsim replay` ran"
              + (f" (last line: {last[:80]})" if last else ""))
    print(f"\nNot uploaded. Before uploading: the tag {tag} must exist on GitHub, or the links "
          "above lead nowhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
