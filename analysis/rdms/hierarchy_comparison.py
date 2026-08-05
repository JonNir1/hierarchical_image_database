"""
Diagnostics comparing every reference RDM against the curated directory hierarchy.

The directory tree under images/<variant>/ *is* the Kiani-Mur semantic hierarchy:
`animate/animal/body/bird/duck3.png` encodes domain -> mid-level -> basic level.
D_sem_km is built from that tree, so it should reproduce it exactly; every other
RDM (WordNet, CLIP, pixels) is an independent measurement whose agreement with
the tree is an empirical question.

The two questions these diagnostics answer are distinct, and are plotted
separately on purpose:

  1. How much rank structure do any two RDMs share?
     -> correlation_matrix() / plot_correlation_matrix()
  2. Does an RDM's distances grow as two images diverge higher up the tree?
     -> depth_profile() / plot_depth_profile()

A high pairwise correlation does not imply a clean depth gradient (an RDM can
track KM well while being flat across the middle of the tree), and a clean
gradient does not imply high correlation (an RDM can order the levels correctly
while disagreeing on every pair within a level). Read them side by side only
after reading them apart.

Note on inference: a condensed RDM has N*(N-1)/2 = 262_450 entries derived from
725 images, so the entries are massively non-independent and the analytic
p-value of any correlation between two RDMs is meaningless. Use
label_shuffle_test(), which permutes image identity (rows and columns together)
and so respects that dependence structure.

The LCA logic here is deliberately a second implementation, independent of the
one inside semantic_km.build_km_rdm(). Sharing it would make
reconstruct_km_from_hierarchy() a tautology instead of a check.

Usage:
    from analysis.rdms import hierarchy_comparison as hc
    hc.correlation_matrix(["sem_km", "clip_pre"])
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from analysis.rdms.common import _EXPECTED_N, load_manifest, load_rdm

# Directory level -> what that level of the tree encodes.
LEVEL_NAMES: dict[int, str] = {
    1: "domain (animate / inanimate)",
    2: "mid-level category",
    3: "basic level",
}

_DEFAULT_RDMS = ["sem_km", "sem_wn", "clip_pre", "clip_post", "sens_pre", "sens_post"]


# ---------------------------------------------------------------------------
# The path hierarchy
# ---------------------------------------------------------------------------


def path_parts() -> list[tuple[str, ...]]:
    """Directory components of every manifest path, in manifest row order."""
    return [
        tuple(Path(p.replace("\\", "/")).parts[:-1])
        for p in load_manifest()["curated_path"]
    ]


def path_depths(parts: list[tuple[str, ...]] | None = None) -> np.ndarray:
    """Number of directory levels below the variant root, per image."""
    parts = path_parts() if parts is None else parts
    return np.array([len(p) for p in parts], dtype=np.float64)


def lca_depth_matrix(parts: list[tuple[str, ...]] | None = None) -> np.ndarray:
    """
    (N, N) depth of each pair's lowest common ancestor directory.

    0 means the two images share no directory at all (they diverge at the root,
    i.e. animate vs inanimate); a larger value means they stay together deeper
    into the tree. The diagonal is zeroed, so read it only off-diagonal.
    """
    parts = path_parts() if parts is None else parts
    n = len(parts)
    lca = np.zeros((n, n), dtype=np.int16)
    for level in range(1, int(path_depths(parts).max()) + 1):
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, p in enumerate(parts):
            if len(p) >= level:
                groups.setdefault(p[:level], []).append(i)
        for members in groups.values():
            if len(members) > 1:
                ix = np.array(members)
                lca[np.ix_(ix, ix)] = level
    np.fill_diagonal(lca, 0)
    return lca


def lca_depth_condensed(parts: list[tuple[str, ...]] | None = None) -> np.ndarray:
    """Condensed (upper-triangle) form of lca_depth_matrix()."""
    return squareform(lca_depth_matrix(parts), checks=False)


def reconstruct_km_from_hierarchy() -> np.ndarray:
    """
    Rebuild D_sem_km from the tree alone: d(i,j) = depth_i + depth_j - 2*lca(i,j),
    floored at 1 off-diagonal.

    Compare against load_rdm("sem_km") to confirm the KM RDM is exactly the tree
    and nothing else. Uses an LCA implementation independent of semantic_km's.
    """
    parts = path_parts()
    depths = path_depths(parts)
    lca = lca_depth_matrix(parts).astype(np.float64)
    dist = np.add.outer(depths, depths) - 2.0 * lca
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 1.0)
    np.fill_diagonal(dist, 0.0)
    return squareform(dist, checks=False)


def depth_is_ragged(parts: list[tuple[str, ...]] | None = None) -> bool:
    """
    True if leaves sit at more than one depth.

    When the tree is ragged, KM distance is not a function of LCA depth alone:
    a pair in a finely-subdivided branch is pushed further apart than an equally
    related pair in a coarse one. That asymmetry propagates into any
    level-stratified comparison, so it is worth knowing about explicitly.
    """
    return len(np.unique(path_depths(parts))) > 1


# ---------------------------------------------------------------------------
# Pairwise agreement between RDMs
# ---------------------------------------------------------------------------


def _z_ranks(d: np.ndarray) -> np.ndarray:
    """Z-scored ranks, so a dot-product mean is a Spearman correlation."""
    r = rankdata(d)
    return (r - r.mean()) / r.std()


def load_rdms(names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load several condensed RDMs by name."""
    return {n: load_rdm(n) for n in (names if names is not None else _DEFAULT_RDMS)}


def correlation_matrix(
    names: list[str] | None = None,
    rdms: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Square DataFrame of Spearman rho between every pair of named RDMs."""
    rdms = load_rdms(names) if rdms is None else rdms
    keys = list(rdms)
    z = {k: _z_ranks(v) for k, v in rdms.items()}
    m = np.array([[float(np.mean(z[a] * z[b])) for b in keys] for a in keys])
    return pd.DataFrame(m, index=keys, columns=keys)


@dataclass
class ShuffleResult:
    """Outcome of an image-label shuffle test."""
    observed: float
    null_mean: float
    null_sd: float
    p_value: float

    @property
    def z(self) -> float:
        return (self.observed - self.null_mean) / self.null_sd

    def __str__(self) -> str:
        return (f"rho={self.observed:+.3f}  null={self.null_mean:+.4f}"
                f"+/-{self.null_sd:.4f}  z={self.z:.1f}  p={self.p_value:.5f}")


def label_shuffle_test(
    a: np.ndarray, b: np.ndarray, *, n_perm: int = 2000, seed: int = 0,
) -> ShuffleResult:
    """
    Test Spearman rho(a, b) against a null that permutes image identity.

    Permuting rows and columns of one RDM together destroys the correspondence
    between the two RDMs while preserving each one's internal dependence
    structure, which a naive entry-wise shuffle would not.

    p is one-sided (P[null >= observed]) with the standard +1 correction, so its
    floor is 1/(n_perm+1).
    """
    rng = np.random.default_rng(seed)
    n = _EXPECTED_N
    za = _z_ranks(a)
    sq = squareform(rankdata(b))
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        r = squareform(sq[np.ix_(p, p)], checks=False)
        null[i] = float(np.mean(za * ((r - r.mean()) / r.std())))
    observed = float(np.mean(za * _z_ranks(b)))
    p_value = (float(np.sum(null >= observed)) + 1.0) / (n_perm + 1.0)
    return ShuffleResult(observed, float(null.mean()), float(null.std()), p_value)


def partial_correlation(a: np.ndarray, b: np.ndarray, control: np.ndarray) -> float:
    """Rank-based partial correlation of a and b, holding `control` fixed."""
    za, zb, zc = _z_ranks(a), _z_ranks(b), _z_ranks(control)
    res_a = za - np.polyval(np.polyfit(zc, za, 1), zc)
    res_b = zb - np.polyval(np.polyfit(zc, zb, 1), zc)
    return float(np.corrcoef(res_a, res_b)[0, 1])


def incremental_r2(
    target: np.ndarray, predictors: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    R^2 of each predictor alone and of all of them jointly, on z-scored ranks.

    If the joint R^2 is close to the sum of the individual ones, the predictors
    carry complementary rather than redundant information about the target.
    """
    zt = _z_ranks(target)
    zp = {k: _z_ranks(v) for k, v in predictors.items()}
    ones = np.ones_like(zt)

    def fit(cols: list[str]) -> float:
        X = np.column_stack([zp[c] for c in cols] + [ones])
        beta, *_ = np.linalg.lstsq(X, zt, rcond=None)
        return float(1.0 - np.var(zt - X @ beta) / np.var(zt))

    rows = [{"predictors": k, "r2": fit([k])} for k in zp]
    rows.append({"predictors": " + ".join(zp), "r2": fit(list(zp))})
    rows.append({"predictors": "(sum if independent)",
                 "r2": sum(r["r2"] for r in rows[:len(zp)])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agreement with the tree specifically
# ---------------------------------------------------------------------------


def depth_profile(
    names: list[str] | None = None,
    rdms: dict[str, np.ndarray] | None = None,
    *, normalise: bool = True,
) -> pd.DataFrame:
    """
    Mean distance at each LCA depth, one column per RDM.

    With normalise=True each column is divided by its value at depth 0, putting
    RDMs with wildly different units (pixel Euclidean vs cosine) on one scale.
    A monotonically decreasing column means that RDM grows with hierarchical
    separation, which is what "tracks the hierarchy" means.
    """
    rdms = load_rdms(names) if rdms is None else rdms
    lca = lca_depth_condensed()
    depths = sorted(np.unique(lca))
    out = pd.DataFrame(index=pd.Index(depths, name="lca_depth"))
    out["n_pairs"] = [int(np.sum(lca == d)) for d in depths]
    for k, v in rdms.items():
        means = np.array([v[lca == d].mean() for d in depths])
        out[k] = means / means[0] if normalise else means
    return out


def level_separation(
    level: int,
    names: list[str] | None = None,
    rdms: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Cohen's d between between-category and within-category distances at one
    directory level.

    Positive d means the RDM places images from different categories further
    apart than images from the same one, i.e. it recovers that level of the
    hierarchy. Images shallower than `level` are excluded from the contrast.
    """
    rdms = load_rdms(names) if rdms is None else rdms
    parts = path_parts()
    valid = np.array([len(p) >= level for p in parts])
    codes = np.array([hash(p[:level]) if len(p) >= level else -1 for p in parts])
    same = squareform((codes[:, None] == codes[None, :]).astype(np.int8),
                      checks=False).astype(bool)
    usable = squareform((valid[:, None] & valid[None, :]).astype(np.int8),
                        checks=False).astype(bool)

    rows = []
    for k, v in rdms.items():
        w, b = v[same & usable], v[(~same) & usable]
        pooled_sd = np.sqrt((w.var() + b.var()) / 2.0)
        gap = b.mean() - w.mean()
        # A degenerate (constant) RDM has no spread to standardise by. Report 0
        # rather than 0/0 = NaN when the group means also coincide, since "no
        # separation" is the honest reading; a non-zero gap with zero spread is
        # genuinely unbounded.
        if pooled_sd == 0:
            d = 0.0 if gap == 0 else np.inf * np.sign(gap)
        else:
            d = gap / pooled_sd
        rows.append({"rdm": k, "within": w.mean(), "between": b.mean(), "cohens_d": d})
    return pd.DataFrame(rows).set_index("rdm")


# ---------------------------------------------------------------------------
# Figures (two questions, two figures, never combined)
# ---------------------------------------------------------------------------

_TEMPLATE = "plotly_white"


def plot_correlation_matrix(corr: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """Heatmap of Spearman rho between RDMs. Answers: what shares rank structure?"""
    labels = [c.replace("_", " ") for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(3).to_numpy(), texttemplate="%{text}",
        textfont={"size": 12}, colorbar=dict(title="&rho;"),
    ))
    fig.update_layout(
        title=title or "Shared rank structure between reference RDMs (Spearman &rho;)",
        template=_TEMPLATE, height=560, width=720,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_depth_profile(profile: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """
    Mean distance against LCA depth. Answers: does distance grow with
    hierarchical separation?
    """
    fig = go.Figure()
    for col in profile.columns:
        if col == "n_pairs":
            continue
        fig.add_trace(go.Scatter(
            x=profile.index, y=profile[col], mode="lines+markers",
            name=col.replace("_", " "),
            line=dict(width=2.5, dash="dash" if col == "sem_km" else "solid"),
            marker=dict(size=9),
        ))
    ticks = [f"{d}<br><sub>n={int(c):,}</sub>"
             for d, c in zip(profile.index, profile["n_pairs"])]
    fig.update_layout(
        title=title or ("Distance vs hierarchical separation"
                        "<br><sub>normalised to each RDM's mean at the root; "
                        "falling curve = tracks the hierarchy</sub>"),
        xaxis=dict(title="LCA depth (0 = diverge at root, higher = stay together deeper)",
                   tickmode="array", tickvals=list(profile.index), ticktext=ticks),
        yaxis=dict(title="mean distance, relative to root"),
        template=_TEMPLATE, height=520, width=860,
        legend=dict(orientation="h", y=-0.24, x=0.5, xanchor="center"),
    )
    return fig
