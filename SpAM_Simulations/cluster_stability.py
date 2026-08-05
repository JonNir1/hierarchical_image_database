"""Do two independent cohorts recover the same cluster structure, and at what granularity?

**Why cluster metrics rather than pair metrics.** The downstream use is stimulus construction:
never put two confusable images in one stimulus. That decision is about groups, not pairs. If three
red flowers look alike, which *pair* among them is "closest" flips with noise, and
``metrics.topk_similar_jaccard`` scores every flip as disagreement even though all of them support
the same practical answer (use one of the three). Global Spearman has the opposite problem: computed
over all 262,450 pairs it is dominated by unrelated-vs-unrelated pairs and reports the easy question.
Cluster agreement is invariant to within-cluster reshuffling, so it measures the decision actually
being made.

**Granularity is the output, not an input.** Clustering two cohorts at every ``k`` and asking where
they still agree tells you the finest resolution the data supports, which is the level at which to
deduplicate. Clusters are discovered **bottom-up** from each cohort's embedding.

**Scope: between-cohort only.** No reference partition is used anywhere in this module. Comparing
recovered clusters against the semantic/path hierarchy is a separate question handled elsewhere;
:func:`partition_agreement` would accept such a partition unchanged, which is exactly why the
boundary is stated rather than left implicit.

**Why VI is primary and ARI is not.** Variation of Information is a true metric on partitions
(Meila 2003): symmetric, non-negative, and obeying the triangle inequality. That is what licenses
chaining two separately-measured claims, ``VI(cohort, ref) <= VI(cohort, cohort') + VI(cohort', ref)``.
ARI is not a distance and does **not** compose, so it must never be substituted into that argument.
ARI and AMI are carried for interpretability only, since VI's units (nats) are unintuitive.

**Why the silhouette ratio.** Raw silhouette is a within-sample fit measure, so it rewards
overfitting; scoring cohort A's labels against cohort B's distances fixes that. It does not fix the
dimensionality confound, because distances concentrate at high D and depress silhouette regardless of
cluster quality. The ``cross / within`` ratio at matched (D, k) cancels that and the small-k bias to
first order, and reads as optimism: near 1 means the separation genuinely reproduces.

**The outcome this must be able to report.** If the space is a continuum rather than lumpy, VI will
be high and flat at every k and no stable granularity exists. That is a finding, not a failure: it
would mean "one image per cluster" is the wrong rule and a distance threshold should be used instead.
High agreement together with near-zero cross-cohort silhouette is the signature of two cohorts
reproducibly agreeing on an arbitrary slicing of a continuum.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, rankdata
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, silhouette_score,
)

DEFAULT_LINKAGES = ("average", "ward", "complete")
DEFAULT_KS = (2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200)
JACCARD_THRESHOLDS = (0.5, 0.75)

# The cross/within ratio needs a denominator that is meaningfully POSITIVE. Below this (including
# any negative value, which means the clustering is worse than useless in-sample) the ratio is
# reported as NaN rather than as an arbitrary or sign-flipped number.
_MIN_WITHIN_SILHOUETTE = 0.02


# --------------------------------------------------------------------------- clustering
def build_linkage(condensed: np.ndarray, method: str = "average") -> np.ndarray:
    """Agglomerative linkage matrix from a condensed distance vector.

    ``method`` selects how the distance between two *clusters* is defined: ``average`` is the mean
    cross-cluster pair distance and assumes least about cluster shape; ``ward`` merges whichever pair
    least increases within-cluster variance, which favours compact equal-sized clusters; ``complete``
    uses the farthest members, giving tight outlier-sensitive clusters.

    Ward is only defined for Euclidean input. That holds here because these vectors come from
    ``pdist`` of a fitted MDS configuration, which is Euclidean by construction - it would *not* hold
    for a raw judged-dissimilarity RDM.
    """
    if method not in DEFAULT_LINKAGES:
        raise ValueError(f"method must be one of {DEFAULT_LINKAGES}, got {method!r}")
    return linkage(condensed, method=method)


def cut_tree(Z: np.ndarray, ks: Sequence[int], n: int) -> Dict[int, np.ndarray]:
    """Cut a dendrogram at each ``k``, returning ``{k: labels}``.

    ``k`` values above ``n`` are dropped with a warning rather than raising, so a shared k grid can
    be reused across image-set sizes.

    Note ``fcluster(criterion="maxclust")`` can return **fewer** than ``k`` clusters when merge
    heights tie, so the realised count is recorded separately and no caller may assume
    ``len(unique(labels)) == k``.
    """
    out: Dict[int, np.ndarray] = {}
    too_large = [k for k in ks if k > n]
    if too_large:
        warnings.warn(f"dropping k values exceeding n_items={n}: {too_large}", stacklevel=2)
    for k in ks:
        if k > n:
            continue
        out[int(k)] = fcluster(Z, t=int(k), criterion="maxclust")
    return out


# --------------------------------------------------------------------------- partition agreement
def _contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Joint counts of the two labelings, as a dense ``(n_a, n_b)`` table."""
    ua, ia = np.unique(labels_a, return_inverse=True)
    ub, ib = np.unique(labels_b, return_inverse=True)
    table = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    return table


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray,
                             normalise: bool = True) -> float:
    """``H(A|B) + H(B|A)`` in nats, optionally divided by ``log(n)``.

    A true metric on the space of partitions (Meila 2003), which is the property that lets two
    separately-measured agreements be chained by the triangle inequality. The ``log(n)`` divisor is a
    constant, so normalising preserves that; most other normalisation schemes do not.

    Bounds: 0 for identical partitions, and exactly ``log(n)`` (so ``1.0`` normalised) between the
    all-in-one-cluster and all-singletons partitions.

    Comparable across ``k`` at fixed ``n``, but **not across different image-set sizes**, since the
    divisor changes with ``n``.
    """
    table = _contingency(labels_a, labels_b)
    n = table.sum()
    if n == 0:
        return np.nan
    joint = table / n
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    # H(A) + H(B) - 2*I(A;B) == H(A|B) + H(B|A), computed in one pass over the non-zero cells.
    mutual = float(np.sum(joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])))
    ha = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = float(-np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    vi = max(ha + hb - 2.0 * mutual, 0.0)      # clamp float error at the identity case
    return vi / np.log(n) if normalise else vi


def partition_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    """VI (raw and normalised), ARI and AMI between two labelings of the same items.

    All four are label-invariant, so the two cohorts' arbitrary cluster numbering does not matter.
    Only VI composes under the triangle inequality; ARI and AMI are for interpretation.
    """
    return {
        "vi": variation_of_information(labels_a, labels_b, normalise=False),
        "vi_norm": variation_of_information(labels_a, labels_b, normalise=True),
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "ami": float(adjusted_mutual_info_score(labels_a, labels_b)),
    }


# --------------------------------------------------------------------------- per-cluster stability
def cluster_wise_jaccard(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """For each cluster in A, its best Jaccard against any cluster in B. One value per A-cluster.

    Returned as a **distribution** rather than a mean because the operational question is *which*
    clusters can be trusted, not whether the partition as a whole is stable. A rock-solid cluster is
    usable even when the rest of the space is mush, and an average would hide exactly that.
    """
    table = _contingency(labels_a, labels_b)
    rows = table.sum(axis=1, keepdims=True)
    cols = table.sum(axis=0, keepdims=True)
    union = rows + cols - table
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, table / union, 0.0)
    return jaccard.max(axis=1)


def jaccard_summary(js: np.ndarray,
                    thresholds: Sequence[float] = JACCARD_THRESHOLDS) -> Dict[str, float]:
    """Central tendency plus the tail of a cluster-wise Jaccard distribution."""
    out: Dict[str, float] = {
        "n_clusters": int(js.size),
        "jaccard_mean": float(js.mean()) if js.size else np.nan,
        "jaccard_median": float(np.median(js)) if js.size else np.nan,
    }
    for t in thresholds:
        key = f"frac_clusters_above_{int(round(t * 100))}"
        out[key] = float((js >= t).mean()) if js.size else np.nan
    return out


# --------------------------------------------------------------------------- separation
def safe_silhouette(square_dist: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette on a precomputed distance matrix, returning NaN instead of raising.

    ``sklearn`` requires ``2 <= n_labels <= n_samples - 1``; both ends of a k grid violate it (k=1,
    and k>=n where every cluster is a singleton), and those cells should be blank rather than fatal.
    """
    n_labels = np.unique(labels).size
    if n_labels < 2 or n_labels >= labels.size:
        return np.nan
    return float(silhouette_score(square_dist, labels, metric="precomputed"))


def silhouette_pair(square_a: np.ndarray, square_b: np.ndarray,
                    labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    """Within- and cross-cohort silhouettes, plus their ratio.

    ``sil_cross_ab`` scores cohort A's cluster assignment against cohort B's geometry, which is the
    out-of-sample question: would a *different* sample of people still see these clusters as
    separated? Both directions are computed and averaged, since the operation is asymmetric.

    ``sil_ratio`` is ``mean(cross) / mean(within)``. Report this rather than the raw cross value: raw
    silhouette falls with dimensionality (distances concentrate) and with k, so it is not comparable
    across the sweep, whereas the ratio shares those biases in numerator and denominator. It is NaN
    when the within value is near zero, where the quotient is meaningless.
    """
    within_a = safe_silhouette(square_a, labels_a)
    within_b = safe_silhouette(square_b, labels_b)
    cross_ab = safe_silhouette(square_b, labels_a)
    cross_ba = safe_silhouette(square_a, labels_b)
    within = np.nanmean([within_a, within_b])
    cross = np.nanmean([cross_ab, cross_ba])
    # The denominator must be meaningfully POSITIVE, not merely non-zero. A negative within-cohort
    # silhouette means points sit closer to another cluster than their own, i.e. there was no
    # separation to lose, and the quotient then flips sign or blows up: on isotropic data at k=12
    # this produced +1.84 from cross=-0.043 over within=-0.023, which reads as "separation improved
    # out of sample". NaN is the honest answer there; `sil_cross` alone still reports the collapse.
    ratio = cross / within if np.isfinite(within) and within >= _MIN_WITHIN_SILHOUETTE else np.nan
    return {
        "sil_within": float(within), "sil_cross": float(cross), "sil_ratio": float(ratio),
        "sil_within_a": within_a, "sil_within_b": within_b,
        "sil_cross_ab": cross_ab, "sil_cross_ba": cross_ba,
    }


# --------------------------------------------------------------------------- whole-tree agreement
def cophenetic_ranks(Z: np.ndarray) -> np.ndarray:
    """Ranked cophenetic distances of a dendrogram (the height at which each pair first merges).

    Pre-ranking here rather than inside the pair loop turns Baker's gamma into a plain Pearson
    correlation, which matters because these vectors are 262,450 long and each pair would otherwise
    re-rank both sides.
    """
    return rankdata(cophenet(Z))


def baker_gamma(ranks_a: np.ndarray, ranks_b: np.ndarray) -> float:
    """Rank correlation between two dendrograms' cophenetic distances. 1 = identical merge order.

    k-free, so it summarises the whole hierarchy rather than one cut. Two caveats: it is dominated by
    the coarse structure, since the overwhelming majority of pairs merge near the root, and it is a
    correlation rather than a metric, so it does **not** compose under the triangle inequality the way
    VI does.
    """
    if ranks_a.size != ranks_b.size or ranks_a.size < 2:
        return np.nan
    return float(pearsonr(ranks_a, ranks_b).statistic)


def cophenetic_fidelity(Z: np.ndarray, condensed: np.ndarray) -> float:
    """How much of a cohort's own geometry its dendrogram retains (Spearman, per fit not per pair).

    A diagnostic on the clustering itself: a low value means the tree is a poor summary of that
    cohort's distances, so agreement between two such trees says little about the geometry.
    """
    return float(pearsonr(rankdata(cophenet(Z)), rankdata(condensed)).statistic)


# --------------------------------------------------------------------------- descriptives
def cluster_size_summary(labels: np.ndarray) -> Dict[str, float]:
    """Shape of a discovered partition. Bottom-up, so none of this can be assumed in advance."""
    _, counts = np.unique(labels, return_counts=True)
    k = counts.size
    p = counts / counts.sum()
    entropy = float(-np.sum(p * np.log(p)))
    return {
        "n_clusters_realised": int(k),
        "size_min": int(counts.min()),
        "size_median": float(np.median(counts)),
        "size_max": int(counts.max()),
        "largest_frac": float(counts.max() / counts.sum()),
        "frac_singletons": float((counts == 1).mean()),
        # 1 = perfectly balanced, -> 0 as one cluster swallows everything.
        "size_entropy_norm": float(entropy / np.log(k)) if k > 1 else np.nan,
    }


# --------------------------------------------------------------------------- one pair of cohorts
def compare_partitions(condensed_a: np.ndarray, condensed_b: np.ndarray,
                       ks: Sequence[int] = DEFAULT_KS,
                       linkages: Sequence[str] = DEFAULT_LINKAGES) -> pd.DataFrame:
    """Every metric for one pair of cohorts, at each (linkage, k). One row per combination.

    The reusable core: it takes two condensed distance vectors and needs no store, so it works
    equally on stored MDS configurations, on a pair of raw RDMs, or on synthetic test data.
    """
    condensed_a = np.asarray(condensed_a, dtype=np.float64)
    condensed_b = np.asarray(condensed_b, dtype=np.float64)
    if condensed_a.shape != condensed_b.shape:
        raise ValueError(f"shape mismatch: {condensed_a.shape} vs {condensed_b.shape}")
    square_a, square_b = squareform(condensed_a), squareform(condensed_b)
    n = square_a.shape[0]

    rows = []
    for method in linkages:
        za, zb = build_linkage(condensed_a, method), build_linkage(condensed_b, method)
        gamma = baker_gamma(cophenetic_ranks(za), cophenetic_ranks(zb))
        cuts_a, cuts_b = cut_tree(za, ks, n), cut_tree(zb, ks, n)
        for k in sorted(cuts_a):
            la, lb = cuts_a[k], cuts_b[k]
            js = cluster_wise_jaccard(la, lb)
            rows.append({
                "linkage": method, "k": int(k),
                **partition_agreement(la, lb),
                **silhouette_pair(square_a, square_b, la, lb),
                **jaccard_summary(js),
                "baker_gamma": gamma,
                # Averaged over the pair: maxclust can realise fewer clusters than requested, and
                # the two cohorts need not tie in the same places.
                "n_clusters_realised": 0.5 * (cluster_size_summary(la)["n_clusters_realised"]
                                              + cluster_size_summary(lb)["n_clusters_realised"]),
            })
    return pd.DataFrame(rows)
