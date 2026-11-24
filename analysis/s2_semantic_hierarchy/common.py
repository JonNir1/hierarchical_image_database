"""
Shared definitions for the s2 notebooks (how similar are the RDMs to the a priori hierarchy?): which
RDMs are compared, how the tree is binned, and the NaN-aware rank agreement every notebook reports.

The RDM set. Six stimulus-derived reference RDMs (sem_km is the tree itself), the cohort-pooled
behavioral RDM (behav_joint, built as D_behav_joint_simple; ~13% of pairs unobserved, so NaN), and from
notebook 1 on two SMACOF embeddings of it (complete), at the dimensionalities notebook 1 selects without
using the tree: behav_joint_mds_low (the stress elbow) and behav_joint_mds_high (the smallest d capturing
95% of the held-out prediction gain).
The per-cohort behavioral RDMs are deliberately absent: s2 is cohort-agnostic, SHINE is s1's question.

Pooling the deep tree. Leaves sit at depth 3, 4 or 5. Depths 4 and 5 are both below the basic level, and
depth 5 is small (73 images), so every analysis pools them into one tier (DEEP_TIER): leaf depth is
binned as {3, >=4}, LCA depth as {0, 1, 2, 3, >=4}, and a pair's branch-depth stratum is the unordered
pair of its two binned leaf depths (3x3, 3x>=4, >=4x>=4).

Usage (from repo root):
    from analysis.s2_semantic_hierarchy import common as c
    rdms = c.load_rdm_set()
    c.rank_agreement(rdms["behav_joint"], rdms[c.REFERENCE])
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.distance import squareform

from analysis.s0_build_rdms.common import REPO_ROOT, RESULTS_DIR, load_rdm, manifest_parts
from analysis.utils.hierarchy import Parts, lca_depth_condensed, path_depths
from analysis.utils.stats import kendall_tau_a, spearman_rho

REFERENCE = "sem_km"
REFERENCE_RDMS = ["sem_km", "sem_wn", "clip_pre", "clip_post", "sens_pre", "sens_post"]
BEHAV = "behav_joint"
BEHAV_MDS_LOW = "behav_joint_mds_low"
BEHAV_MDS_HIGH = "behav_joint_mds_high"
BEHAV_MDS_SET = [BEHAV_MDS_LOW, BEHAV_MDS_HIGH]
RAW_SET = [*REFERENCE_RDMS, BEHAV]
FULL_SET = [*RAW_SET, *BEHAV_MDS_SET]
RDM_FILE = {BEHAV: "behav_joint_simple"}
#: What each reference RDM is expected to be against the tree: `sem_km` is the tree itself (the ceiling),
#: the pixel RDMs are the low-level floor. An expectation used to frame results, not an assumption of any test.
EXPECTED_ROLE = {"sem_km": "ceiling", "sens_pre": "floor", "sens_post": "floor"}

S2_RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "semantic_hierarchy"

LEVEL_NAMES: dict[int, str] = {1: "domain (animate / inanimate)", 2: "mid-level category", 3: "basic level"}
DEEP_TIER = 4

RANK_STATS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "kendall_a": kendall_tau_a, "spearman": spearman_rho,
}
STAT_LABEL = {"kendall_a": "Kendall tau-a", "spearman": "Spearman rho"}


# ---------------------------------------------------------------------------
# The RDM set
# ---------------------------------------------------------------------------


def cache_path(name: str) -> Path:
    """Location of one s2 cache file (bootstraps, nulls, MDS fits); gitignored with analysis/results/."""
    return S2_RESULTS_DIR / name


def load_rdm_set(names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Condensed RDMs by s2 name (default FULL_SET), in the order given."""
    names = FULL_SET if names is None else names
    missing = [n for n in BEHAV_MDS_SET if n in names and not (RESULTS_DIR / f"D_{n}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not built yet: run 1_mds_dimensionality.ipynb first, or pass names=RAW_SET."
        )
    return {n: load_rdm(RDM_FILE.get(n, n)) for n in names}


# ---------------------------------------------------------------------------
# Pooled tree bins
# ---------------------------------------------------------------------------


def tier_label(depth: int) -> str:
    return f">={DEEP_TIER}" if depth >= DEEP_TIER else str(depth)


def pooled_leaf_depths(parts: Parts) -> np.ndarray:
    """Leaf depth per image with every depth >= DEEP_TIER set to DEEP_TIER."""
    return np.minimum(path_depths(parts), DEEP_TIER).astype(int)


def lca_bins(parts: Parts) -> np.ndarray:
    """Condensed LCA depth per pair with every depth >= DEEP_TIER set to DEEP_TIER."""
    return np.minimum(lca_depth_condensed(parts), DEEP_TIER).astype(int)


def lca_bin_labels(bins: np.ndarray) -> list[str]:
    """Axis labels for the sorted unique values of lca_bins()."""
    return [tier_label(int(b)) for b in np.unique(bins)]


def branch_strata(parts: Parts) -> np.ndarray:
    """Condensed branch-depth stratum label per pair, e.g. '3x3', '3x>=4', '>=4x>=4'."""
    depths = pooled_leaf_depths(parts)
    lo = np.minimum(depths[:, None], depths[None, :])
    hi = np.maximum(depths[:, None], depths[None, :])
    lo_c = squareform(lo, checks=False)
    hi_c = squareform(hi, checks=False)
    labels = {(a, b): f"{tier_label(a)}x{tier_label(b)}" for a in np.unique(depths) for b in np.unique(depths)}
    return np.array([labels[(a, b)] for a, b in zip(lo_c, hi_c)], dtype=object)


def tree_parts() -> Parts:
    """The curated tree, in manifest order (a seam tests monkeypatch with a toy tree)."""
    return manifest_parts()


# ---------------------------------------------------------------------------
# NaN-aware rank agreement
# ---------------------------------------------------------------------------


def pairwise_complete(*rdms: np.ndarray) -> np.ndarray:
    """Boolean mask of the pairs observed (non-NaN) in every RDM given."""
    return np.logical_and.reduce([~np.isnan(r) for r in rdms])


def rank_agreement(a: np.ndarray, b: np.ndarray, stats: tuple[str, ...] = tuple(RANK_STATS)) -> dict[str, float]:
    """Each rank statistic between `a` and `b` over their pairwise-complete pairs."""
    mask = pairwise_complete(a, b)
    return {k: float(RANK_STATS[k](a[mask], b[mask])) for k in stats}


def rank_agreement_list(a: np.ndarray, b: np.ndarray) -> list[float]:
    """rank_agreement() as a list in RANK_STATS order, the shape utils.stats.image_bootstrap() expects."""
    return list(rank_agreement(a, b).values())
