"""
Which levels of the tree does each RDM recover? (4_level_by_level.ipynb is the front end.)

At tree level L (common.LEVEL_NAMES: 1 domain, 2 mid-level, 3 basic) an RDM "recovers" the level if
images of different categories are further apart than images of the same category. The effect size is
Cohen's d between those two sets of distances (positive = recovered). Pre-registered H1b: the effect is
strongest at the superordinate levels and weakest at the basic level (Mur et al., 2013).

Two contrasts, with the same d:
  global  between = every pair in different level-L categories; within = pairs in the same one.
  nested  between = pairs that share their level L-1 ancestor but differ at level L; within as above.
          For L = 1 the parent is the root, so nested equals global.
The global "between" set at the basic level also holds animate vs inanimate pairs, which any RDM
separates easily, so it inflates the deep levels. The nested contrast compares siblings only.

Method choices. Pairs involving an image shallower than L are excluded. Everything is NaN-aware
(unobserved pairs of the raw behavioral RDM are dropped). Cohen's d uses the pooled population SD of the
two groups, and returns 0 for a constant RDM with equal group means. Uncertainty is the image bootstrap
(stimuli are the sampling unit): per-pair label arrays ("same category at L") are resampled together with
the distances, so group membership stays consistent. H1b's ordering is tested on bootstrap CIs of
d_L - d_(L+1), with the fraction of draws in the predicted direction (positive).

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import level_separation as ls
    ls.nested_separation_table(rdms, n_boot=1000, cache_dir=c.cache_path(""))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import squareform

from analysis.s2_semantic_hierarchy import common as c
from analysis.utils import plotting
from analysis.utils.hierarchy import Parts
from analysis.utils.stats import image_bootstrap, percentile_ci

_TEMPLATE = "plotly_white"
LEVELS = list(c.LEVEL_NAMES)
CONTRASTS = ("nested", "global")


# ---------------------------------------------------------------------------
# Pair labels and the contrast
# ---------------------------------------------------------------------------


def same_category(parts: Parts, level: int) -> np.ndarray:
    """Condensed float indicator: 1 if two images share their level-`level` category, 0 if not, NaN if
    either image is shallower than `level`."""
    keys = [p[:level] if len(p) >= level else None for p in parts]
    codes = {k: i for i, k in enumerate(sorted({k for k in keys if k is not None}))}
    code = np.array([codes.get(k, -1) for k in keys])
    known = code >= 0
    same = (code[:, None] == code[None, :]).astype(float)
    same[~(known[:, None] & known[None, :])] = np.nan
    return squareform(same, checks=False)


def cohens_d(within: np.ndarray, between: np.ndarray) -> float:
    """Cohen's d of between minus within (pooled population SD); NaN if either group is empty."""
    if within.size == 0 or between.size == 0:
        return float("nan")
    pooled_sd = np.sqrt((within.var() + between.var()) / 2.0)
    gap = between.mean() - within.mean()
    # No spread to standardise by: report 0 for equal means, +-inf for a real gap.
    if pooled_sd == 0:
        return 0.0 if gap == 0 else float(np.inf * np.sign(gap))
    return float(gap / pooled_sd)


def _groups(d: np.ndarray, same: np.ndarray, parent_same: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(within, between) distances; `parent_same` (nested contrast) restricts between to siblings."""
    observed = ~np.isnan(d)
    within = observed & (same == 1)
    between = observed & (same == 0)
    if parent_same is not None:
        between &= parent_same == 1
    return d[within], d[between]


def _all_levels(d: np.ndarray, *same: np.ndarray) -> list[float]:
    """d at every level: global levels first, then nested levels (the bootstrap statistic)."""
    out = [cohens_d(*_groups(d, same[i], None)) for i in range(len(same))]
    out += [cohens_d(*_groups(d, same[i], same[i - 1] if i else None)) for i in range(len(same))]
    return out


def level_separation(level: int, rdms: dict[str, np.ndarray], *, nested: bool = False) -> pd.DataFrame:
    """Within and between mean distance and Cohen's d at one level, one row per RDM (index `rdm`)."""
    parts = c.tree_parts()
    same = same_category(parts, level)
    parent = same_category(parts, level - 1) if nested and level > 1 else None
    rows = []
    for name, d in rdms.items():
        w, b = _groups(d, same, parent)
        rows.append({"rdm": name, "within": w.mean() if w.size else np.nan,
                     "between": b.mean() if b.size else np.nan, "cohens_d": cohens_d(w, b),
                     "n_within": w.size, "n_between": b.size})
    return pd.DataFrame(rows).set_index("rdm")


# ---------------------------------------------------------------------------
# Bootstrap: d per level, global and nested
# ---------------------------------------------------------------------------


def bootstrap_draws(
    rdms: dict[str, np.ndarray], *, n_boot: int, cache_dir: Path | None, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Image-bootstrap draws per RDM, shape (n_boot, 2 * n_levels): global d per level, then nested d."""
    labels = [same_category(c.tree_parts(), level) for level in LEVELS]
    draws = {}
    for name, d in rdms.items():
        path = None if cache_dir is None else Path(cache_dir) / f"level_separation_{name}.npz"
        draws[name] = image_bootstrap(
            [d, *labels], _all_levels, n_boot=n_boot, seed=seed, cache_path=path, extra_key=("level_separation",),
        )
    return draws


def point_estimates(rdms: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    labels = [same_category(c.tree_parts(), level) for level in LEVELS]
    return {name: np.array(_all_levels(d, *labels)) for name, d in rdms.items()}


def separation_table(rdms: dict[str, np.ndarray], *, n_boot: int, cache_dir: Path | None) -> pd.DataFrame:
    """Long table: rdm, contrast, level, d, ci_lo, ci_hi (image-bootstrap 95% percentile CI)."""
    draws = bootstrap_draws(rdms, n_boot=n_boot, cache_dir=cache_dir)
    point = point_estimates(rdms)
    n = len(LEVELS)
    rows = []
    for name in rdms:
        for k, contrast in enumerate(CONTRASTS):
            for i, level in enumerate(LEVELS):
                lo, hi = percentile_ci(draws[name][:, k * n + i][np.isfinite(draws[name][:, k * n + i])])
                rows.append({"rdm": name, "contrast": "global" if k == 0 else "nested", "level": level,
                             "d": point[name][k * n + i], "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def nested_separation_table(rdms: dict[str, np.ndarray], *, n_boot: int, cache_dir: Path | None) -> pd.DataFrame:
    """Nested d per level with bootstrap CIs, for an arbitrary dict of condensed RDMs (rdm, level, d, ci_lo, ci_hi)."""
    table = separation_table(rdms, n_boot=n_boot, cache_dir=cache_dir)
    return table[table["contrast"] == "nested"].drop(columns="contrast").reset_index(drop=True)


def ordering_table(rdms: dict[str, np.ndarray], *, n_boot: int, cache_dir: Path | None) -> pd.DataFrame:
    """H1b ordering per RDM and contrast: d_L - d_(L+1) with its bootstrap CI and the fraction of draws > 0."""
    draws = bootstrap_draws(rdms, n_boot=n_boot, cache_dir=cache_dir)
    point = point_estimates(rdms)
    n = len(LEVELS)
    rows = []
    for name in rdms:
        for k, contrast in enumerate(CONTRASTS):
            for i in range(n - 1):
                diff = draws[name][:, k * n + i] - draws[name][:, k * n + i + 1]
                diff = diff[np.isfinite(diff)]
                lo, hi = percentile_ci(diff)
                rows.append({
                    "rdm": name, "contrast": "global" if k == 0 else "nested",
                    "step": f"d{LEVELS[i]} - d{LEVELS[i + 1]}",
                    "diff": point[name][k * n + i] - point[name][k * n + i + 1],
                    "ci_lo": lo, "ci_hi": hi, "frac_predicted": float(np.mean(diff > 0)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptive: per basic-level category
# ---------------------------------------------------------------------------


def category_separation(rdms: dict[str, np.ndarray], *, level: int = 3) -> pd.DataFrame:
    """Nested d per level-`level` category (rows) and RDM (columns), descriptive only.

    Within = pairs inside the category; between = pairs of one of its images with an image of a sibling
    category (same level L-1 ancestor, different level-L category). NaN where either group is empty.
    """
    parts = c.tree_parts()
    keys = [p[:level] if len(p) >= level else None for p in parts]
    cats = sorted({k for k in keys if k is not None})
    n = len(parts)
    same = squareform(same_category(parts, level), checks=False)
    parent = squareform(same_category(parts, level - 1), checks=False) if level > 1 else np.ones((n, n))
    iu = np.triu_indices(n, 1)
    out = pd.DataFrame(index=pd.Index(["/".join(k) for k in cats], name="category"), columns=list(rdms), dtype=float)
    for name, d in rdms.items():
        square = squareform(d, checks=False)
        for cat in cats:
            member = np.array([k == cat for k in keys])
            inside = member[:, None] & member[None, :]
            sibling = member[:, None] ^ member[None, :]
            sibling &= (parent == 1) & (same == 0)
            w = square[inside & ~np.eye(n, dtype=bool)]
            b = square[sibling]
            out.loc["/".join(cat), name] = cohens_d(w[~np.isnan(w)], b[~np.isnan(b)])
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_levels(table: pd.DataFrame) -> go.Figure:
    """d by level per RDM, nested (left) and global (right), CI as error bars."""
    names = plotting.rdm_legend_order(list(table["rdm"].unique()))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Nested contrast (siblings only)", "Global contrast (all other categories)"))
    for col, contrast in enumerate(("nested", "global"), start=1):
        for k, name in enumerate(names):
            sub = table[(table["rdm"] == name) & (table["contrast"] == contrast)].sort_values("level")
            offset = (k - (len(names) - 1) / 2) * 0.07
            fig.add_trace(go.Scatter(
                x=sub["level"] + offset, y=sub["d"], mode="lines+markers", showlegend=col == 1,
                error_y={"type": "data", "symmetric": False, "array": sub["ci_hi"] - sub["d"],
                         "arrayminus": sub["d"] - sub["ci_lo"], "thickness": 1.2},
                **plotting.rdm_scatter_kwargs(name, size=9, width=1.8),
            ), row=1, col=col)
    ticks = {"tickmode": "array", "tickvals": LEVELS, "ticktext": [c.LEVEL_NAMES[lv] for lv in LEVELS]}
    fig.update_xaxes(**ticks)
    fig.update_yaxes(title_text="Cohen's d (between vs within)", row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=520, width=1100, title="Separation of tree levels by RDM (95% image-bootstrap CI)")
    return fig


def fig_categories(table: pd.DataFrame) -> go.Figure:
    """Heatmap of category_separation(): categories x RDMs."""
    names = plotting.rdm_legend_order(list(table.columns))
    z = table[names].to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(len(names))), y=list(table.index), colorscale="RdBu", zmid=0,
        text=np.round(z, 2), texttemplate="%{text}", colorbar={"title": "d"},
    ))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(names))), ticktext=[plotting.rdm_tick_label(n) for n in names])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template=_TEMPLATE, height=640, width=900,
                      title="Nested separation per basic-level category (d, category vs its siblings)")
    return fig
