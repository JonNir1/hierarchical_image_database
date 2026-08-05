"""
How far is what a vision model sees from what the curated hierarchy says?

For every image we build two independent views of "what this thing is" and
measure the WordNet shortest-path distance between them:

  ImageNet view : the top-3 ResNet-50 predictions, each an ImageNet-1K class,
                  each of which is itself a WordNet synset.
  Path view     : three concept slots read off the curated path,
                  images/<variant>/<...>/<dirname>/<filename>NN.png
                    - "filename"  the filename stem with trailing digits stripped
                                  (tennis3.png -> tennis)
                    - "dirname"   the immediate parent directory (ball)
                    - "combined"  the two as one compound concept, tried in both
                                  orders, so ball/tennis3.png reaches
                                  "tennis ball" and not only "ball tennis"

That gives a 3x3 grid of distances per image. We record the whole grid, its
minimum, and which cell produced that minimum, because the argmin is the
interesting part: it says whether the model's first guess was usable, and
whether the filename or the folder carries the concept.

Each concept slot can be polysemous ("bird" has several noun synsets), so a
slot's distance is the minimum over its own synsets, and we keep the winning
synset for inspection.

Why this is a diagnostic and not an RDM builder: ImageNet-1K contains no
person, face, or human-body categories, so all 164 human images in this dataset
are unclassifiable in principle and their scores reflect the nearest available
non-human class. D_sem_wn is therefore built by direct label-to-synset
assignment (see semantic_wn_dir.py and assign_wn_synsets.py). This module
measures the gap between the two views rather than trying to close it, and the
human images are the clearest illustration of why the ResNet route was dropped.

Inputs:
    analysis/rdms/imagenet_class_index.json   tracked; maps class idx -> wnid.
                                              Needed because torchvision ships
                                              class names but not wnids.
Outputs (to analysis/results/rdms/):
    imagenet_vs_path.csv   one row per image: full grid, min, argmin, synsets

Usage (from repo root):
    python -m analysis.rdms.imagenet_vs_path
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from analysis.rdms.common import (
    RESULTS_DIR, image_paths, load_manifest, open_as_rgb_pil,
)

CLASS_INDEX_PATH = Path(__file__).parent / "imagenet_class_index.json"
TABLE_PATH = RESULTS_DIR / "imagenet_vs_path.csv"

#: Path concept slots, in the grid's row order.
PATH_SLOTS: tuple[str, ...] = ("filename", "dirname", "combined")
TOP_K = 3

#: Distance used when a synset pair has no connecting path. Matches
#: semantic_wn_dir._WN_FALLBACK_DIST so the two modules stay comparable.
WN_FALLBACK_DIST = 30.0


# ---------------------------------------------------------------------------
# ImageNet class index -> WordNet
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_class_index() -> dict[str, list[str]]:
    """Map str(class_idx) -> [wnid, class_name] for the 1000 ImageNet classes."""
    if not CLASS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{CLASS_INDEX_PATH} not found. It is tracked in the repo; restore it "
            "rather than re-downloading, so the class ordering is guaranteed to "
            "match the one the cached predictions were built against."
        )
    with open(CLASS_INDEX_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def imagenet_synset(class_idx: int) -> "wn.Synset | None":
    """WordNet synset for an ImageNet class index, or None if unresolvable."""
    wnid, _ = load_class_index()[str(class_idx)]
    try:
        return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
    except (WordNetError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Path concepts
# ---------------------------------------------------------------------------


def path_tokens(curated_path: str) -> tuple[str, str]:
    """(filename stem without trailing digits, immediate parent directory)."""
    p = Path(curated_path.replace("\\", "/"))
    stem = re.sub(r"\d+$", "", p.stem).lower()
    parent = p.parts[-2].lower() if len(p.parts) >= 2 else ""
    return stem, parent


def _synsets_for_phrase(phrase: str) -> list["wn.Synset"]:
    """Noun synsets for a phrase, trying both spaced and underscored forms."""
    phrase = phrase.strip()
    if not phrase:
        return []
    found = list(wn.synsets(phrase, pos=wn.NOUN))
    found += [s for s in wn.synsets(phrase.replace(" ", "_"), pos=wn.NOUN)
              if s not in found]
    return found


def path_concept_synsets(curated_path: str) -> dict[str, list["wn.Synset"]]:
    """
    Candidate noun synsets for each of the three path slots.

    "combined" pools both compound orderings, since which one is correct depends
    on the folder: ball/tennis -> "tennis ball", but bird/song -> "songbird".
    A slot maps to an empty list when no noun synset exists for it.
    """
    stem, parent = path_tokens(curated_path)
    return {
        "filename": _synsets_for_phrase(stem),
        "dirname": _synsets_for_phrase(parent),
        "combined": (_synsets_for_phrase(f"{parent} {stem}")
                     + _synsets_for_phrase(f"{stem} {parent}")),
    }


# ---------------------------------------------------------------------------
# The 3 x 3 grid
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _wn_distance(a: "wn.Synset", b: "wn.Synset") -> float:
    d = a.shortest_path_distance(b)
    return float(d) if d is not None else WN_FALLBACK_DIST


def _slot_distance(
    slot_synsets: list["wn.Synset"], target: "wn.Synset | None",
) -> tuple[float, str | None]:
    """
    Best distance from a polysemous slot to one ImageNet synset.

    Returns (nan, None) when the slot has no synsets or the target is
    unresolvable, so that empty cells are excluded from the argmin rather than
    competing with real ones.
    """
    if target is None or not slot_synsets:
        return float("nan"), None
    best_d, best_s = float("inf"), None
    for s in slot_synsets:
        d = _wn_distance(s, target)
        if d < best_d:
            best_d, best_s = d, s
    return best_d, (best_s.name() if best_s is not None else None)


def distance_grid(
    curated_path: str, top_k_indices: list[int],
) -> tuple[np.ndarray, list[list[str | None]]]:
    """
    (3, K) distance grid: rows are PATH_SLOTS, columns are ResNet ranks.

    Returns (distances, winning_slot_synset_names). NaN marks a cell that could
    not be scored.
    """
    slots = path_concept_synsets(curated_path)
    grid = np.full((len(PATH_SLOTS), len(top_k_indices)), np.nan)
    names: list[list[str | None]] = [[None] * len(top_k_indices) for _ in PATH_SLOTS]
    for c, idx in enumerate(top_k_indices):
        target = imagenet_synset(idx)
        for r, slot in enumerate(PATH_SLOTS):
            grid[r, c], names[r][c] = _slot_distance(slots[slot], target)
    return grid, names


# ---------------------------------------------------------------------------
# ResNet-50 top-K
# ---------------------------------------------------------------------------


def classify_top_k(variant: str = "pre_shine", k: int = TOP_K) -> np.ndarray:
    """
    ResNet-50 top-k ImageNet class indices per image, shape (N, k).

    Runs on the pre-SHINE images by default: the question is what the model
    makes of the stimulus, and SHINE is a separate manipulation.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    transform = weights.transforms()

    paths = image_paths(variant)
    print(f"[imagenet_vs_path] ResNet-50 top-{k} over {len(paths)} '{variant}' images ...")
    out = np.empty((len(paths), k), dtype=np.int64)
    for i, p in enumerate(tqdm(paths, desc=f"resnet50_{variant}")):
        tensor = transform(open_as_rgb_pil(p)).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
        out[i] = logits.topk(k, dim=1).indices.squeeze(0).numpy()
    return out


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def build_comparison_table(
    variant: str = "pre_shine", *, save: bool = True,
) -> pd.DataFrame:
    """
    One row per image: the full 3x3 grid, its minimum, and the winning cell.

    Columns include d_<slot>_r<rank> for all nine cells, plus min_distance,
    best_slot, best_rank, and the two synsets that produced the minimum.
    """
    manifest = load_manifest()
    top_k = classify_top_k(variant)
    class_index = load_class_index()

    rows = []
    for path, topk in tqdm(
        zip(manifest["curated_path"], top_k), total=len(manifest), desc="scoring grid",
    ):
        grid, names = distance_grid(path, list(topk))
        rec: dict = {"curated_path": path}
        for r, slot in enumerate(PATH_SLOTS):
            for c in range(grid.shape[1]):
                rec[f"d_{slot}_r{c + 1}"] = grid[r, c]

        if np.all(np.isnan(grid)):
            rec.update(min_distance=np.nan, best_slot=None, best_rank=np.nan,
                       best_path_synset=None, best_imagenet_synset=None,
                       best_imagenet_class=None, scorable_cells=0)
        else:
            r, c = np.unravel_index(np.nanargmin(grid), grid.shape)
            syn = imagenet_synset(int(topk[c]))
            rec.update(
                min_distance=float(grid[r, c]),
                best_slot=PATH_SLOTS[r],
                best_rank=int(c) + 1,
                best_path_synset=names[r][c],
                best_imagenet_synset=syn.name() if syn else None,
                best_imagenet_class=class_index[str(int(topk[c]))][1],
                scorable_cells=int(np.sum(~np.isnan(grid))),
            )
        rec["top1_class"] = class_index[str(int(topk[0]))][1]
        rec["manifest_synset"] = manifest.loc[len(rows), "wn_synset_name"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLE_PATH, index=False)
        print(f"[imagenet_vs_path] wrote {TABLE_PATH}  ({len(df)} rows)")
    return df


def load_comparison_table() -> pd.DataFrame:
    """Load the cached table, or raise directing the caller to build it."""
    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"{TABLE_PATH} not found. Run `python -m analysis.rdms.imagenet_vs_path`."
        )
    return pd.read_csv(TABLE_PATH)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def slot_rank_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of which of the nine cells won, path slot by ResNet rank."""
    return pd.crosstab(df["best_slot"], df["best_rank"]).reindex(
        index=[s for s in PATH_SLOTS], fill_value=0,
    )


def _branch(curated_path: str, levels: int = 2) -> str:
    return "/".join(curated_path.replace("\\", "/").split("/")[:levels])


def _is_human(curated_path: str) -> bool:
    return curated_path.replace("\\", "/").startswith("animate/human")


def _share_a_lemma(a: str, b: str) -> bool:
    """True if two synsets share a lemma, i.e. they differ only in word sense."""
    try:
        la = {l.name().lower() for l in wn.synset(a).lemmas()}
        lb = {l.name().lower() for l in wn.synset(b).lemmas()}
    except (WordNetError, ValueError):
        return False
    return bool(la & lb)


def disagreement_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per image where the grid's path synset differs from the manifest's
    curated synset, with how far apart they are and whether the difference is
    pure polysemy.

    Reading `best_path_synset`: it is not "what the path says", it is the path
    sense that minimises distance to ResNet. The grid actively selects whichever
    sense flatters the model, so agreement here is an upper bound on how often
    automatic parsing would land on the curated concept if asked to pick the
    *correct* sense rather than the most convenient one.
    """
    scorable = df["best_path_synset"].notna() & df["manifest_synset"].notna()
    sub = df.loc[scorable & (df["best_path_synset"] != df["manifest_synset"]),
                 ["curated_path", "best_path_synset", "manifest_synset"]].copy()
    sub["same_lemma"] = [_share_a_lemma(a, b) for a, b in
                         zip(sub["best_path_synset"], sub["manifest_synset"])]
    sub["wn_distance"] = [_wn_distance(wn.synset(a), wn.synset(b)) for a, b in
                          zip(sub["best_path_synset"], sub["manifest_synset"])]
    sub["is_human"] = sub["curated_path"].map(_is_human)
    sub["branch"] = sub["curated_path"].map(_branch)
    return sub


def disagreement_summary(df: pd.DataFrame, *, by: str = "branch") -> pd.DataFrame:
    """
    Agreement between the grid's path synset and the manifest's curated synset,
    grouped by 'branch' (top two path levels) or 'human'.

    A low agreement rate means automatic path parsing would have produced a
    different concept than the curated assignment for most images in that group.
    """
    if by not in ("branch", "human"):
        raise ValueError(f"by must be 'branch' or 'human', got {by!r}")
    work = df.copy()
    work["group"] = (work["curated_path"].map(_is_human).map({True: "human", False: "non-human"})
                     if by == "human" else work["curated_path"].map(_branch))
    scorable = work["best_path_synset"].notna() & work["manifest_synset"].notna()
    agree = scorable & (work["best_path_synset"] == work["manifest_synset"])

    out = pd.DataFrame({
        "n": work.groupby("group").size(),
        "n_scorable": scorable.groupby(work["group"]).sum(),
        "n_agree": agree.groupby(work["group"]).sum(),
    })
    out["n_disagree"] = out["n_scorable"] - out["n_agree"]
    out["agree_pct"] = (out["n_agree"] / out["n_scorable"].replace(0, np.nan) * 100).round(1)
    return out.sort_values("n", ascending=False)


def slot_marginal_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-slot best distance ignoring the other slots, so the slots can be
    compared as if each were used alone.
    """
    rows = []
    for slot in PATH_SLOTS:
        cols = [f"d_{slot}_r{r + 1}" for r in range(TOP_K)]
        best = df[cols].min(axis=1)
        rows.append({
            "slot": slot,
            "n_scorable": int(best.notna().sum()),
            "mean": float(best.mean()),
            "median": float(best.median()),
            "n_exact": int((best == 0).sum()),
        })
    return pd.DataFrame(rows).set_index("slot")


# ---------------------------------------------------------------------------
# Figures (one question each)
# ---------------------------------------------------------------------------

_TEMPLATE = "plotly_white"


def plot_distance_by_rank(df: pd.DataFrame) -> go.Figure:
    """Minimum distance, split by which ResNet rank supplied it."""
    fig = go.Figure()
    labels = {1: "rank 1 (top-1 was best)", 2: "rank 2 (top-1 overridden)",
              3: "rank 3 (top-2 overridden)"}
    for rank in sorted(df["best_rank"].dropna().unique()):
        sub = df.loc[df["best_rank"] == rank, "min_distance"].dropna()
        fig.add_trace(go.Histogram(
            x=sub, name=f"{labels.get(int(rank), rank)} (n={len(sub)})",
            xbins=dict(size=1), opacity=0.75,
        ))
    fig.update_layout(
        title="WordNet distance from ImageNet concept to path concept, by winning rank",
        xaxis_title="shortest WordNet path distance", yaxis_title="images",
        barmode="overlay", template=_TEMPLATE, height=480, width=880,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    return fig


def plot_distance_by_slot(df: pd.DataFrame) -> go.Figure:
    """Minimum distance, split by which path slot supplied it."""
    fig = go.Figure()
    for slot in PATH_SLOTS:
        sub = df.loc[df["best_slot"] == slot, "min_distance"].dropna()
        fig.add_trace(go.Histogram(
            x=sub, name=f"{slot} (n={len(sub)})", xbins=dict(size=1), opacity=0.75,
        ))
    fig.update_layout(
        title="WordNet distance from ImageNet concept to path concept, by winning path slot",
        xaxis_title="shortest WordNet path distance", yaxis_title="images",
        barmode="overlay", template=_TEMPLATE, height=480, width=880,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    return fig


def plot_slot_rank_heatmap(df: pd.DataFrame) -> go.Figure:
    """Counts over the 3x3 grid: which cell won, how often."""
    ct = slot_rank_counts(df)
    fig = go.Figure(go.Heatmap(
        z=ct.to_numpy(), x=[f"rank {c}" for c in ct.columns], y=list(ct.index),
        colorscale="Blues", text=ct.to_numpy(), texttemplate="%{text}",
        textfont={"size": 14}, colorbar=dict(title="images"),
    ))
    fig.update_layout(
        title="Which of the nine (path slot x ResNet rank) cells gave the closest match",
        template=_TEMPLATE, height=420, width=680,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_agreement_by_branch(df: pd.DataFrame) -> go.Figure:
    """
    Agreement with the curated synset per branch, as stacked counts.

    The branch that collapses is the argument for manual assignment.
    """
    s = disagreement_summary(df, by="branch")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.index, y=s["n_agree"], name="matches curated synset",
                         marker_color="#2ca02c",
                         text=[f"{p:.0f}%" for p in s["agree_pct"]], textposition="inside"))
    fig.add_trace(go.Bar(x=s.index, y=s["n_disagree"], name="differs from curated synset",
                         marker_color="#d62728"))
    fig.update_layout(
        title=("Would automatic path parsing have reproduced the curated synset?"
               "<br><sub>per branch; the human branch is why the manual assignment "
               "in assign_wn_synsets.py is the reference</sub>"),
        xaxis_title="branch", yaxis_title="images", barmode="stack",
        template=_TEMPLATE, height=480, width=880,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    return fig


if __name__ == "__main__":
    table = build_comparison_table()
    print("\n=== winning cell counts (path slot x ResNet rank) ===")
    print(slot_rank_counts(table).to_string())
    print("\n=== per-slot marginal distance ===")
    print(slot_marginal_distance(table).round(2).to_string())
    print(f"\nmedian minimum distance: {table['min_distance'].median():.1f}")
    print(f"exact matches (distance 0): {int((table['min_distance'] == 0).sum())}")

    print("\n=== agreement with the curated synset, by branch ===")
    print(disagreement_summary(table, by="branch").to_string())
    print("\n=== agreement with the curated synset, human vs not ===")
    print(disagreement_summary(table, by="human").to_string())

    detail = disagreement_detail(table)
    print(f"\n=== {len(detail)} disagreements ===")
    print(f"  pure polysemy (same lemma, different sense): "
          f"{detail['same_lemma'].mean():.1%}")
    print(f"  median WordNet distance between the two:     "
          f"{detail['wn_distance'].median():.1f}")
    print("\n  most common disagreeing pairs:")
    print(detail.groupby(["best_path_synset", "manifest_synset"])
          .agg(n=("curated_path", "size"), wn_distance=("wn_distance", "first"))
          .sort_values("n", ascending=False).head(10).to_string())
