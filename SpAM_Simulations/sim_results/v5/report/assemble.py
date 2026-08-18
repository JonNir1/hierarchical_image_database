"""Build the shareable v5 report: hand-written prose plus computed figures.

    python SpAM_Simulations/sim_results/v5/report/assemble.py --run <path to the v5 run dir>

``report_v5.src.html`` holds the prose and is edited by hand. Wherever it contains

    <!-- figure: coverage_by_n -->

that comment is replaced by the rendered figure. The build fails if the source names a figure that
does not exist, or if a registered figure is never used, so the two cannot drift apart silently.

A second placeholder inlines a static image from the repository:

    <!-- image: SpAM_Task/assets/examples/before1.png | a trial as it opens -->

The path is resolved against the repository root and the file is embedded as a base64 data URI, so
the built page stays a single shareable file with no external references.

The run's data (``out/``, ``mds_store/``, ``gt/``, ``gt_diagnostics/``) is gitignored and may live in
a different checkout from this file, which is why ``--run`` exists. The built page is written next to
the source and is itself gitignored, being ~5 MB with plotly inlined.
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))

import plotly.io as pio  # noqa: E402
from plotly.offline import get_plotlyjs  # noqa: E402

from figures import FIGURES, Run  # noqa: E402

PLACEHOLDER = re.compile(r"[ \t]*<!--\s*figure:\s*([a-z0-9_]+)\s*-->[ \t]*\n?")
IMAGE = re.compile(r"<!--\s*image:\s*([^|>]+?)\s*(?:\|\s*(.*?)\s*)?-->")

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}

CSS = """
:root { color-scheme: light dark; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       line-height: 1.6; color: #1a1a1a; background: #fff; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 2rem 1.5rem 5rem; }
h1 { font-size: 1.8rem; margin: 0 0 .3rem; }
h2 { font-size: 1.3rem; margin: 2.75rem 0 .6rem; padding-top: .9rem;
     border-top: 1px solid rgba(128,128,128,.28); }
h3 { font-size: 1rem; margin: 1.6rem 0 .4rem; color: #444; }
.sub { color: #666; margin: 0 0 1.5rem; font-size: .93rem; }
p { margin: .7rem 0; }
ul, ol { margin: .7rem 0; padding-left: 1.3rem; }
li { margin: .35rem 0; }
code { background: rgba(128,128,128,.14); padding: .1em .35em; border-radius: 3px; font-size: .9em; }
.bottom-line { border-left: 3px solid #1f77b4; background: rgba(31,119,180,.07);
               padding: .7rem 1rem; margin: 1rem 0; border-radius: 0 4px 4px 0; font-size: .97rem; }
.figcaption { font-size: .8rem; color: #666; margin: -.4rem 0 1.6rem; padding: 0 .3rem;
              line-height: 1.45; }
.warn { border-left: 3px solid #d62728; background: rgba(214,39,40,.07);
        padding: .7rem 1rem; margin: 1rem 0; border-radius: 0 4px 4px 0; }
table { border-collapse: collapse; margin: 1rem 0; font-size: .86rem; width: 100%;
        display: block; overflow-x: auto; }
th, td { padding: .35rem .65rem; text-align: right; border-bottom: 1px solid rgba(128,128,128,.2); }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid rgba(128,128,128,.4); font-weight: 600; }
table.kv th { text-align: left; width: 32%; font-weight: 600; }
nav { background: rgba(128,128,128,.08); padding: .9rem 1.2rem; border-radius: 6px;
      margin: 1.2rem 0 2rem; font-size: .92rem; }
nav ol { margin: .3rem 0; }
a { color: #1f77b4; }
.example-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .35rem 1rem;
                margin: 1.2rem 0 .2rem; align-items: start; }
.example-grid .colhead { margin: 0; font-size: .82rem; font-weight: 600; color: #666;
                         text-transform: uppercase; letter-spacing: .04em; }
img.example { width: 100%; height: auto; display: block; border-radius: 4px;
              border: 1px solid rgba(128,128,128,.35); }
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e8; background: #14161a; }
  .sub, h3, .figcaption, .example-grid .colhead { color: #9aa0a6; }
  a { color: #6db3f2; }
}
"""


def embed_images(source: str) -> str:
    """Inline every ``<!-- image: path | alt -->`` as a self-contained data URI."""
    count = 0

    def swap(match: re.Match) -> str:
        nonlocal count
        rel, alt = match.group(1), (match.group(2) or "")
        path = REPO_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"[assemble] ABORT: the source references a missing image {rel!r} "
                             f"(looked under {REPO_ROOT}).")
        mime = MIME.get(path.suffix.lower())
        if mime is None:
            raise SystemExit(f"[assemble] ABORT: unsupported image type {path.suffix!r} for {rel!r}.")
        uri = base64.b64encode(path.read_bytes()).decode("ascii")
        count += 1
        return f'<img class="example" alt="{alt}" src="data:{mime};base64,{uri}">'

    out = IMAGE.sub(swap, source)
    print(f"[assemble] inlined {count} static images")
    return out


def render(source: str, run: Run) -> str:
    """Replace every figure placeholder, and report any that do not line up."""
    used: List[str] = []
    unknown: List[str] = []

    def swap(match: re.Match) -> str:
        name = match.group(1)
        if name not in FIGURES:
            unknown.append(name)
            return match.group(0)
        used.append(name)
        spec = FIGURES[name]
        html = pio.to_html(spec.fn(run), include_plotlyjs=False, full_html=False,
                           config={"displayModeBar": False})
        # The caption travels with the figure rather than the prose, so a figure can never appear
        # without the reader being told what its error bars are.
        caption = f'<p class="figcaption"><em>{spec.caption}</em></p>'
        return f"{html}\n{caption}\n"

    out = PLACEHOLDER.sub(swap, source)
    if unknown:
        raise SystemExit(f"[assemble] ABORT: the source references unknown figure(s) "
                         f"{sorted(set(unknown))}. Register them in figures.FIGURES.")
    unused = sorted(set(FIGURES) - set(used))
    if unused:
        raise SystemExit(f"[assemble] ABORT: {len(unused)} registered figure(s) never appear in the "
                         f"source: {unused}.\n  Either place them, or remove them from "
                         f"figures.FIGURES so the two stay in step.")
    print(f"[assemble] placed {len(used)} figures")
    return out


def build_page(body: str, title: str) -> str:
    toc = "".join(
        f'<li><a href="#{anchor}">{name}</a></li>'
        for anchor, name in re.findall(r'<section id="([^"]+)"><h2>(?:\d+\.\s*)?([^<]+)</h2>', body))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
<script>{get_plotlyjs()}</script>
</head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">Simulating the SpAM study: how many participants we need, how to allocate images to
trials, and what the resulting embeddings can support.</p>
<nav><strong>Contents</strong><ol>{toc}</ol></nav>
{body}
</div></body></html>"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=HERE.parent,
                   help="the v5 run directory holding out/, mds_store/, gt/ and gt_diagnostics/. "
                        "Defaults to this file's parent, which is right once the report and the "
                        "data live in the same checkout.")
    p.add_argument("--src", type=Path, default=HERE / "report_v5.src.html")
    p.add_argument("--out", type=Path, default=HERE / "report_v5.html")
    p.add_argument("--title", default="SpAM Simulations v5")
    args = p.parse_args(argv)

    if not (args.run / "out").is_dir():
        raise SystemExit(
            f"[assemble] ABORT: no out/ under {args.run}.\n"
            f"  The run data is gitignored and may sit in another checkout; pass it explicitly:\n"
            f"    --run <repo>/SpAM_Simulations/sim_results/v5")

    source = embed_images(args.src.read_text(encoding="utf-8"))
    page = build_page(render(source, Run(args.run)), args.title)
    args.out.write_text(page, encoding="utf-8")
    print(f"[assemble] wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
