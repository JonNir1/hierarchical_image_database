"""
Stimulus-level manipulation check (analysis/s1_pre_vs_post_shine/0_sensory.ipynb): how much does SHINE
change the pixel and CLIP RDMs? No behavioral data and no reference RDM (such as the Kiani tree) is
involved: the pre-SHINE RDM of a feature space is compared with the post-SHINE RDM of the same space.

Three views, from coarse to fine:

  rdm_agreement / bootstrap_rdm_agreement -- how similar are the two RDMs as wholes? Kendall tau-a and
      Spearman rho (rank structure), Pearson r (linear agreement on the raw distances) and the ratio of
      mean distances (did the scale shift?). Both RDMs are deterministic functions of the 725 images, so
      the only uncertainty is which images were sampled: the CI is an image bootstrap (images resampled
      with replacement, pairs of an image with its own duplicate dropped), which respects the dependence
      between the 262,450 pairs that an entry-wise resampling would ignore.
  row_agreement -- per image, how well does its row of the pre RDM (its distances to all other images)
      rank-agree with its row of the post RDM?
  feature_shift -- per image, how far did the feature vector itself move between pre and post
      (cosine distance and Euclidean distance)? For CLIP these are the stored embeddings; for pixels the
      flattened RGB images, streamed from disk.

Usage (from repo root):
    from analysis.s1_pre_vs_post_shine import sensory
    boot = sensory.bootstrap_rdm_agreement(load_rdm("clip_pre"), load_rdm("clip_post"))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from analysis.s0_build_rdms.common import (
    REPO_ROOT, image_paths, load_embeddings, load_image_rgb, load_manifest, order_hash,
)
from analysis.s1_pre_vs_post_shine.reference_agreement import percentile_ci
from analysis.utils.disk_cache import cached_array, cached_frame, fingerprint
from analysis.utils.stats import kendall_tau_a

SENSORY_CACHE_DIR = REPO_ROOT / "analysis" / "results" / "pre_vs_post_shine" / "sensory_cache"

STATS = ("kendall_a", "spearman", "pearson", "scale_ratio")
STAT_LABEL = {
    "kendall_a": "Kendall tau-a", "spearman": "Spearman rho", "pearson": "Pearson r",
    "scale_ratio": "mean distance, post / pre",
}
SPACE_LABEL = {"sens": "pixels (Euclidean)", "clip": "CLIP ViT-B/32 (cosine)"}
_TEMPLATE = "plotly_white"
_SPACE_COLOR = {"sens": "#c8692b", "clip": "#2b7bb9"}


# ---------------------------------------------------------------------------
# RDM-level agreement
# ---------------------------------------------------------------------------


def _agreement_stats(pre: np.ndarray, post: np.ndarray) -> list[float]:
    return [
        kendall_tau_a(pre, post),
        float(np.corrcoef(rankdata(pre), rankdata(post))[0, 1]),
        float(np.corrcoef(pre, post)[0, 1]),
        float(post.mean() / pre.mean()),
    ]


def rdm_agreement(d_pre: np.ndarray, d_post: np.ndarray) -> dict[str, float]:
    """Point estimates of STATS between two condensed RDMs over the same images."""
    return dict(zip(STATS, _agreement_stats(d_pre, d_post)))


def bootstrap_rdm_agreement(
    d_pre: np.ndarray, d_post: np.ndarray, *, n_boot: int = 1000, seed: int = 0,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Image-bootstrap draws of STATS, one row per draw.

    Each draw resamples the images with replacement, then recomputes the statistics over the pairs of
    the resampled set; a pair of an image with its own duplicate (distance 0 in both RDMs) is dropped,
    so duplicates add repeated pairs but no artificial agreement. `cache_path` stores the draws.
    """
    def draw() -> np.ndarray:
        pre_sq, post_sq = squareform(d_pre), squareform(d_post)
        n = len(pre_sq)
        iu = np.triu_indices(n, 1)
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot, len(STATS)))
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            i, j = idx[iu[0]], idx[iu[1]]
            keep = i != j
            draws[b] = _agreement_stats(pre_sq[i[keep], j[keep]], post_sq[i[keep], j[keep]])
        return draws

    draws = cached_array(cache_path, fingerprint(d_pre, d_post, n_boot, seed), draw, key="draws")
    return pd.DataFrame(draws, columns=list(STATS))


def agreement_table(
    d_pre: np.ndarray, d_post: np.ndarray, boot: pd.DataFrame, *, level: float = 0.95,
) -> pd.DataFrame:
    """Point estimate and percentile image-bootstrap CI of every statistic."""
    estimate = rdm_agreement(d_pre, d_post)
    rows = []
    for stat in STATS:
        lo, hi = percentile_ci(boot[stat].to_numpy(), level)
        rows.append({"statistic": STAT_LABEL[stat], "estimate": estimate[stat], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("statistic")


# ---------------------------------------------------------------------------
# Per-image views
# ---------------------------------------------------------------------------


def row_agreement(d_pre: np.ndarray, d_post: np.ndarray) -> np.ndarray:
    """Spearman rho, per image, between its row of the pre RDM and its row of the post RDM (diagonal
    excluded). NaN for an image whose row is constant."""
    pre_sq, post_sq = squareform(d_pre), squareform(d_post)
    n = len(pre_sq)
    off_diagonal = ~np.eye(n, dtype=bool)

    def z_ranks(sq: np.ndarray) -> np.ndarray:
        ranks = rankdata(sq[off_diagonal].reshape(n, n - 1), axis=1)
        return ranks - ranks.mean(axis=1, keepdims=True)

    z_pre, z_post = z_ranks(pre_sq), z_ranks(post_sq)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (z_pre * z_post).sum(axis=1) / np.sqrt((z_pre ** 2).sum(axis=1) * (z_post ** 2).sum(axis=1))


def shift_between(pre: np.ndarray, post: np.ndarray) -> pd.DataFrame:
    """Row-wise cosine and Euclidean distance between matched (N, D) feature matrices."""
    pre, post = pre.astype(np.float64), post.astype(np.float64)
    norms = np.linalg.norm(pre, axis=1) * np.linalg.norm(post, axis=1)
    return pd.DataFrame({
        "cosine": 1.0 - (pre * post).sum(axis=1) / norms,
        "euclidean": np.linalg.norm(pre - post, axis=1),
    })


def _with_image_column(shift: pd.DataFrame) -> pd.DataFrame:
    shift.insert(0, "curated_path", load_manifest()["curated_path"].to_numpy())
    return shift


def clip_shift() -> pd.DataFrame:
    """Per-image cosine and Euclidean distance between the pre and post CLIP embeddings."""
    return _with_image_column(shift_between(load_embeddings("clip_pre"), load_embeddings("clip_post")))


def pixel_shift(cache_path: Path | None = None) -> pd.DataFrame:
    """Per-image cosine and Euclidean distance between the pre and post flattened RGB images
    (streamed from disk, one image pair at a time)."""
    def compute() -> pd.DataFrame:
        rows = []
        for pre_path, post_path in zip(image_paths("pre_shine"), image_paths("post_shine")):
            pair = np.stack([load_image_rgb(p).ravel() for p in (pre_path, post_path)])
            rows.append(shift_between(pair[:1], pair[1:]).iloc[0])
        return _with_image_column(pd.DataFrame(rows).reset_index(drop=True))

    return cached_frame(cache_path, fingerprint(order_hash(), "pixel_shift"), compute)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_agreement_forest(tables: dict[str, pd.DataFrame]) -> go.Figure:
    """Point estimate and CI of the three agreement statistics, one row per feature space."""
    stats = [STAT_LABEL[s] for s in STATS[:3]]
    fig = go.Figure()
    for space, table in tables.items():
        sub = table.loc[stats]
        fig.add_trace(go.Scatter(
            x=sub["estimate"], y=stats, mode="markers", name=SPACE_LABEL[space],
            marker=dict(size=11, color=_SPACE_COLOR[space]),
            error_x=dict(type="data", symmetric=False, array=sub["ci_high"] - sub["estimate"],
                         arrayminus=sub["estimate"] - sub["ci_low"], color=_SPACE_COLOR[space]),
        ))
    fig.update_layout(
        title="Pre- vs post-SHINE RDM agreement (95% image-bootstrap CI)", template=_TEMPLATE,
        xaxis=dict(title="agreement between the pre and post RDM", range=[0, 1.02]),
        yaxis=dict(autorange="reversed"), height=380, width=820,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )
    return fig


def fig_pre_post_density(rdms: dict[str, tuple[np.ndarray, np.ndarray]], *, bins: int = 80) -> go.Figure:
    """Pre vs post distance of every image pair (log-count heatmap), with the identity line."""
    fig = make_subplots(rows=1, cols=len(rdms), subplot_titles=[SPACE_LABEL[s] for s in rdms],
                        horizontal_spacing=0.12)
    for col, (space, (d_pre, d_post)) in enumerate(rdms.items(), start=1):
        counts, x_edges, y_edges = np.histogram2d(d_pre, d_post, bins=bins)
        fig.add_trace(go.Heatmap(
            z=np.log10(1.0 + counts.T), x=x_edges[:-1], y=y_edges[:-1], colorscale="Viridis",
            colorbar=dict(title="log10(1 + pairs)", x=1.0 if col == len(rdms) else None, len=0.85),
            showscale=col == len(rdms),
        ), row=1, col=col)
        top = float(max(d_pre.max(), d_post.max()))
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", showlegend=False,
                                 line=dict(color="white", dash="dash", width=1.5)), row=1, col=col)
        fig.update_xaxes(title_text="distance, pre-SHINE", row=1, col=col)
        fig.update_yaxes(title_text="distance, post-SHINE", row=1, col=col)
    fig.update_layout(title="Every image pair, before vs after SHINE", template=_TEMPLATE,
                      height=480, width=420 * len(rdms) + 120)
    return fig


def fig_shift_vs_row_agreement(per_image: dict[str, pd.DataFrame]) -> go.Figure:
    """Per image: how far its feature vector moved (cosine) against how well its RDM row survived."""
    fig = make_subplots(rows=1, cols=len(per_image), subplot_titles=[SPACE_LABEL[s] for s in per_image],
                        horizontal_spacing=0.12)
    for col, (space, frame) in enumerate(per_image.items(), start=1):
        fig.add_trace(go.Scatter(
            x=frame["cosine"], y=frame["row_rho"], mode="markers", showlegend=False,
            marker=dict(size=5, opacity=0.5, color=_SPACE_COLOR[space]),
            customdata=frame["curated_path"], hovertemplate="%{customdata}<br>shift %{x:.4f}<br>row rho %{y:.3f}",
        ), row=1, col=col)
        fig.update_xaxes(title_text="cosine distance, pre vs post vector", row=1, col=col)
        fig.update_yaxes(title_text="row rho, pre vs post RDM", row=1, col=col)
    fig.update_layout(title="Which images did SHINE move, and did their neighbourhoods survive?",
                      template=_TEMPLATE, height=460, width=420 * len(per_image) + 80)
    return fig
