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
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, rankdata
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, silhouette_score,
)
from tqdm.auto import tqdm

DEFAULT_LINKAGES = ("average", "ward", "complete")

# The granularity grid. 725 images, so k=200 averages 3.6 images per cluster and k=150 averages 4.8.
#
# READ THE TOP OF THIS GRID WITH CARE. k >= 150 cuts at roughly leaf granularity, and the pilot the
# ground truth was fitted from supports very little structure at that scale: within `same_leaf`
# pairs the raw data's own half-split reliability is about 0.05 (see gt_diagnostics), so a GT built
# on it is largely interpolating there. Agreement scores at those k are therefore reporting on the
# embedding's smoothness as much as on recoverable structure, and should not carry the same weight
# as k <= 50. They are kept rather than dropped because the *trend* across k is informative and
# truncating the grid would hide where it breaks down - but a `k_star` landing at 150 or 200 is a
# statement about the ground truth, not a recommendation.
HIGH_K_THRESHOLD = 150
DEFAULT_KS = (2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200)

# Rep pairs sampled per group rather than exhausted; see `sample_pairs` for the trade.
DEFAULT_MAX_PAIRS = 22
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
                    labels_a: np.ndarray, labels_b: np.ndarray,
                    within_a: Optional[float] = None,
                    within_b: Optional[float] = None) -> Dict[str, float]:
    """Within- and cross-cohort silhouettes, plus their ratio.

    ``sil_cross_ab`` scores cohort A's cluster assignment against cohort B's geometry, which is the
    out-of-sample question: would a *different* sample of people still see these clusters as
    separated? Both directions are computed and averaged, since the operation is asymmetric.

    ``sil_ratio`` is ``mean(cross) / mean(within)``. Report this rather than the raw cross value: raw
    silhouette falls with dimensionality (distances concentrate) and with k, so it is not comparable
    across the sweep, whereas the ratio shares those biases in numerator and denominator. It is NaN
    when the within value is near zero, where the quotient is meaningless.
    """
    # The WITHIN values depend only on (fit, linkage, k) - not on the pairing - so a caller walking
    # C(n, 2) pairs recomputes each of them once per pair the fit appears in. At 10 reps and 22
    # sampled pairs that is ~4.4x more silhouette sweeps than necessary, and silhouette on a
    # precomputed 725x725 matrix is the single most expensive operation in this module. Pass them in
    # from `_PreparedFit.within` to skip the repeats; omitting them keeps the standalone behaviour.
    within_a = safe_silhouette(square_a, labels_a) if within_a is None else within_a
    within_b = safe_silhouette(square_b, labels_b) if within_b is None else within_b
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


# --------------------------------------------------------------------------- store-level drivers
def _require_conf(store) -> None:
    """Clustering needs coordinates, so a confdist-only store cannot serve it."""
    if not store.stores_conf:
        raise ValueError(
            "this store has no MDS configurations, so cohorts cannot be clustered. Re-run the sweep "
            "with `run_mds_sweep(..., store_conf=True)` (the default); stores written before "
            "configurations were recorded support only the distance-vector metrics."
        )


class _PreparedFit:
    """One cohort's clustering, computed once so the O(reps^2) pair loop stays cheap.

    Linkage, cutting and cophenetic ranking are O(reps); only the comparisons are O(reps^2). Doing
    them inside the pair loop instead would rebuild each fit's tree ``n_reps - 1`` times.
    """

    __slots__ = ("condensed", "square", "trees", "labels", "coph_ranks", "within")

    def __init__(self, condensed: np.ndarray, ks: Sequence[int], linkages: Sequence[str]):
        self.condensed = condensed
        self.square = squareform(condensed)
        n = self.square.shape[0]
        self.trees = {m: build_linkage(condensed, m) for m in linkages}
        self.labels = {m: cut_tree(z, ks, n) for m, z in self.trees.items()}
        self.coph_ranks = {m: cophenetic_ranks(z) for m, z in self.trees.items()}
        # Computed once per (linkage, k) rather than once per pair this fit takes part in.
        self.within = {(m, k): safe_silhouette(self.square, lab)
                       for m, by_k in self.labels.items() for k, lab in by_k.items()}


def _fit_distances(store, row: int, ndim: int) -> np.ndarray:
    """Condensed distances of one stored configuration.

    Recomputed from ``conf`` rather than read from ``confdist``: the two are equal, but coordinates
    are ~20x smaller on disk, so a conf-only download is the normal input here. It also guarantees
    the vector is Euclidean in ``ndim`` dimensions, which is what makes Ward linkage well-defined.
    """
    return pdist(np.asarray(store.conf(int(row), int(ndim)), dtype=np.float64))


def _prepare_group(store, rows: Sequence[int], ndim: int, ks: Sequence[int],
                   linkages: Sequence[str]) -> List["_PreparedFit"]:
    return [_PreparedFit(_fit_distances(store, r, ndim), ks, linkages) for r in rows]


_PAIR_METRICS = ("vi", "vi_norm", "ari", "ami", "sil_within", "sil_cross", "sil_ratio",
                 "jaccard_mean", "jaccard_median", "frac_clusters_above_50",
                 "frac_clusters_above_75", "n_clusters_realised")


def _mean_sem(values: Sequence[float], name: str) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return {
        f"mean_{name}": float(vals.mean()) if vals.size else np.nan,
        f"sem_{name}": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
    }


def sample_pairs(n_fits: int, max_pairs: Optional[int] = DEFAULT_MAX_PAIRS,
                 seed: int = 0) -> List[tuple]:
    """Up to ``max_pairs`` of the C(n_fits, 2) rep pairs, drawn once per group.

    At ``reps=10`` the full set is 45 pairs and the pair loop dominates the whole analysis. Halving
    it to 22 buys roughly a 40% cut in wall clock for a sqrt(45/22) ~ 1.4x widening of the SEM,
    which is a good trade here: these pairs were never independent to begin with (each cohort
    appears in ``n_reps - 1`` of them), so the reported SEM understates the true uncertainty either
    way, and the quantity being estimated is a mean over a large grid of configurations.

    Drawn once per group and reused across every (linkage, k), so metrics within a group are
    computed on the same cohort pairs and stay comparable. ``max_pairs=None`` restores the full set.
    """
    pairs = list(combinations(range(n_fits), 2))
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=int(max_pairs), replace=False)
    return [pairs[i] for i in sorted(idx)]


def _agreement_rows(fits: Sequence["_PreparedFit"], ks: Sequence[int], linkages: Sequence[str],
                    max_pairs: Optional[int] = DEFAULT_MAX_PAIRS,
                    pair_seed: int = 0) -> List[Dict[str, float]]:
    """Mean and SEM of every metric over the sampled rep pairs, per (linkage, k)."""
    rows: List[Dict[str, float]] = []
    pairs = sample_pairs(len(fits), max_pairs, pair_seed)
    for method in linkages:
        for k in sorted(set(ks) & set(fits[0].labels[method])):
            acc: Dict[str, List[float]] = {m: [] for m in _PAIR_METRICS}
            for i, j in pairs:
                fa, fb = fits[i], fits[j]
                la, lb = fa.labels[method][k], fb.labels[method][k]
                scores = {
                    **partition_agreement(la, lb),
                    **silhouette_pair(fa.square, fb.square, la, lb,
                                      within_a=fa.within.get((method, k)),
                                      within_b=fb.within.get((method, k))),
                    **jaccard_summary(cluster_wise_jaccard(la, lb)),
                    "n_clusters_realised": 0.5 * (np.unique(la).size + np.unique(lb).size),
                }
                for m in _PAIR_METRICS:
                    acc[m].append(scores[m])
            row: Dict[str, float] = {"linkage": method, "k": int(k),
                                     "n_reps": len(fits), "n_pairs": len(pairs)}
            for m in _PAIR_METRICS:
                row.update(_mean_sem(acc[m], m))
            rows.append(row)
    return rows


def _agreement_worker(store_path, base, rows, ndim, ks, linkages, max_pairs, pair_seed):
    """One configuration group's agreement rows. Top-level so it can be pickled to a worker."""
    from SpAM_Simulations.pipeline import _worker_store

    fits = _prepare_group(_worker_store(store_path), rows, ndim, ks, linkages)
    return [{**base, **row}
            for row in _agreement_rows(fits, ks, linkages, max_pairs, pair_seed)]


def compute_cluster_agreement(store, ks: Sequence[int] = DEFAULT_KS,
                              linkages: Sequence[str] = DEFAULT_LINKAGES,
                              group_fields=None, verbose: bool = True, n_jobs: int = -1,
                              max_pairs: Optional[int] = DEFAULT_MAX_PAIRS,
                              pair_seed: int = 0) -> pd.DataFrame:
    """Between-cohort cluster agreement per (configuration, ndim, linkage, k).

    Mirrors ``pipeline.compute_embedding_generalizability``: reps within a group are independently
    simulated cohorts, so the rep-pair comparisons answer "would a second run of this study recover
    the same clusters?".

    Note the pairs are **not independent** - each cohort appears in many of them - so the reported
    SEM understates the true uncertainty. Every rep-pair metric in the pipeline shares that
    limitation; it is stated here rather than inherited quietly, and it is also why sampling
    ``max_pairs`` of them rather than exhausting C(n_reps, 2) costs less than it appears to.

    Groups are independent, so they are the parallel axis. ``n_jobs=1`` runs in-process.
    """
    from SpAM_Simulations.pipeline import group_tasks, map_groups

    _require_conf(store)
    tasks = group_tasks(store, group_fields, min_size=2)
    return pd.DataFrame(map_groups(
        store, tasks, _agreement_worker, n_jobs=n_jobs, desc="Cluster agreement", verbose=verbose,
        ks=tuple(ks), linkages=tuple(linkages), max_pairs=max_pairs, pair_seed=pair_seed))


def _dendrogram_worker(store_path, base, rows, ndim, linkages, max_pairs, pair_seed):
    """One group's k-free tree agreement. Top-level so it can be pickled to a worker."""
    from SpAM_Simulations.pipeline import _worker_store

    fits = _prepare_group(_worker_store(store_path), rows, ndim, (), linkages)
    pairs = sample_pairs(len(fits), max_pairs, pair_seed)
    out = []
    for method in linkages:
        gammas = [baker_gamma(fits[i].coph_ranks[method], fits[j].coph_ranks[method])
                  for i, j in pairs]
        fidelity = [cophenetic_fidelity(f.trees[method], f.condensed) for f in fits]
        out.append({**base, "linkage": method, "n_reps": len(fits), "n_pairs": len(gammas),
                    **_mean_sem(gammas, "baker_gamma"),
                    **_mean_sem(fidelity, "cophenetic_fidelity")})
    return out


def compute_dendrogram_agreement(store, linkages: Sequence[str] = DEFAULT_LINKAGES,
                                 group_fields=None, verbose: bool = True, n_jobs: int = -1,
                                 max_pairs: Optional[int] = DEFAULT_MAX_PAIRS,
                                 pair_seed: int = 0) -> pd.DataFrame:
    """k-free agreement between cohorts' whole dendrograms, per (configuration, ndim, linkage).

    Baker's gamma summarises the entire merge structure rather than one cut. ``cophenetic_fidelity``
    reports how much of each cohort's own geometry its tree retains, which qualifies the first: if
    the trees are poor summaries of their own distances, their agreeing says little about the space.
    Fidelity is per-fit, so it is unaffected by ``max_pairs``; Baker's gamma is per-pair and is.
    """
    from SpAM_Simulations.pipeline import group_tasks, map_groups

    _require_conf(store)
    tasks = group_tasks(store, group_fields, min_size=2)
    return pd.DataFrame(map_groups(
        store, tasks, _dendrogram_worker, n_jobs=n_jobs, desc="Dendrogram agreement",
        verbose=verbose, linkages=tuple(linkages), max_pairs=max_pairs, pair_seed=pair_seed))


_SIZE_FIELDS = ("n_clusters_realised", "size_min", "size_median", "size_max", "largest_frac",
                "frac_singletons", "size_entropy_norm")


# --------------------------------------------------------------------------- choosing a granularity
DEFAULT_GROUP_BY = ("num_subjects", "ndim", "linkage")

# A VI range narrower than this across the whole k grid means no granularity is distinguishably
# better than any other, i.e. there is no k* to find.
FLAT_VI_TOLERANCE = 0.02
# Below this, cross-cohort silhouette says the "clusters" are not separated in the other cohort's
# geometry, however well the two agree on where to cut.
ARBITRARY_SLICING_SILHOUETTE = 0.05


def select_k(agreement: pd.DataFrame, criterion: str = "vi_norm", rule: str = "one_se",
             by: Sequence[str] = DEFAULT_GROUP_BY) -> pd.DataFrame:
    """Choose a granularity per group. One row per group with ``k_star``.

    Reuses ``gt_construction.select_ndim``'s one-SE rule on the ``k`` axis rather than
    reimplementing it, so dimensionality and granularity are chosen by the same logic: take the
    **smallest** k whose mean is within one standard error of the best. On a flat curve a plain
    argmax is noise-driven and drifts to fine granularities, and for a deduplication rule the
    parsimonious end is the safe one - a coarser k merges more images and excludes more candidate
    pairs, which is the conservative error.
    """
    from SpAM_Simulations.gt_construction import _HIGHER_IS_BETTER, apply_selection_rule

    by = [c for c in by if c in agreement.columns]
    mean_col, sem_col = f"mean_{criterion}", f"sem_{criterion}"
    if mean_col not in agreement.columns:
        raise ValueError(f"agreement frame has no {mean_col!r}; got {sorted(agreement.columns)}")
    if criterion not in _HIGHER_IS_BETTER:
        raise ValueError(f"unknown criterion {criterion!r}; add it to gt_construction._HIGHER_IS_BETTER")

    rows = []
    for key, grp in agreement.groupby(by, dropna=False):
        # This frame already carries mean_/sem_ columns, so the rule is applied to them directly.
        # Re-summarising would take a SEM over one value per k, giving NaN and quietly turning the
        # one-SE rule into a plain argmax.
        grp = grp.sort_values("k")
        k_star = apply_selection_rule(
            grp["k"].to_numpy(), grp[mean_col].to_numpy(),
            grp[sem_col].to_numpy() if sem_col in grp.columns else np.full(len(grp), np.nan),
            higher_is_better=_HIGHER_IS_BETTER[criterion], rule=rule,
        )
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key_tuple)), "k_star": int(k_star),
                     "criterion": criterion, "rule": rule})
    return pd.DataFrame(rows)


def continuum_diagnostics(agreement: pd.DataFrame, criterion: str = "vi_norm",
                          by: Sequence[str] = DEFAULT_GROUP_BY,
                          flat_tol: float = FLAT_VI_TOLERANCE,
                          sil_tol: float = ARBITRARY_SLICING_SILHOUETTE) -> pd.DataFrame:
    """Does a stable, *meaningful* granularity exist? One row per group, with two verdicts.

    Both verdicts are **findings, not errors**, and the analysis must be able to report them rather
    than silently returning a k\\*:

    * ``is_flat`` - the criterion varies by less than ``flat_tol`` across the entire k grid, so no
      granularity is distinguishably better and k\\* is arbitrary.
    * ``is_arbitrary_slicing`` - cross-cohort silhouette at k\\* is below ``sil_tol``, so the cohorts
      reproducibly agree on a cut of a space that has no separation in it. Agreement without
      separation is the signature of a continuum.

    Either verdict means "one image per cluster" is the wrong rule for this data, and a distance
    threshold should be used instead.
    """
    by = [c for c in by if c in agreement.columns]
    mean_col = f"mean_{criterion}"
    chosen = select_k(agreement, criterion=criterion, by=by).set_index(by)["k_star"]

    rows = []
    for key, grp in agreement.groupby(by, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        k_star = int(chosen.loc[key])
        at_star = grp[grp["k"] == k_star]
        vals = grp[mean_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        vi_range = float(vals.max() - vals.min()) if vals.size else np.nan
        sil_cross = float(at_star["mean_sil_cross"].iloc[0]) if len(at_star) else np.nan
        sil_ratio = float(at_star["mean_sil_ratio"].iloc[0]) if len(at_star) else np.nan
        rows.append({
            **dict(zip(by, key_tuple)),
            "k_star": k_star,
            f"{criterion}_at_k_star": float(at_star[mean_col].iloc[0]) if len(at_star) else np.nan,
            f"{criterion}_range": vi_range,
            "sil_cross_at_k_star": sil_cross,
            "sil_ratio_at_k_star": sil_ratio,
            "is_flat": bool(np.isfinite(vi_range) and vi_range < flat_tol),
            "is_arbitrary_slicing": bool(np.isfinite(sil_cross) and sil_cross < sil_tol),
        })
    return pd.DataFrame(rows)


def _sizes_worker(store_path, base, rows, ndim, ks, linkages):
    """One group's partition-shape rows. Per-fit, so no pair sampling applies."""
    from SpAM_Simulations.pipeline import _worker_store

    fits = _prepare_group(_worker_store(store_path), rows, ndim, ks, linkages)
    if not fits:
        return []
    out = []
    for method in linkages:
        for k in sorted(set(ks) & set(fits[0].labels[method])):
            summaries = [cluster_size_summary(f.labels[method][k]) for f in fits]
            row = {**base, "linkage": method, "k": int(k), "n_reps": len(fits)}
            for field in _SIZE_FIELDS:
                row.update(_mean_sem([s[field] for s in summaries], field))
            out.append(row)
    return out


def compute_cluster_sizes(store, ks: Sequence[int] = DEFAULT_KS,
                          linkages: Sequence[str] = DEFAULT_LINKAGES,
                          group_fields=None, verbose: bool = True,
                          n_jobs: int = -1) -> pd.DataFrame:
    """Shape of the discovered partitions, averaged over reps, per (configuration, ndim, linkage, k).

    Descriptive rather than evaluative, and necessary precisely because the clusters are found
    bottom-up: their number and size distribution cannot be assumed in advance, and an agreement
    score is hard to read without knowing whether the partition is balanced or one giant cluster
    plus dust.
    """
    from SpAM_Simulations.pipeline import group_tasks, map_groups

    _require_conf(store)
    tasks = group_tasks(store, group_fields, min_size=1)
    return pd.DataFrame(map_groups(
        store, tasks, _sizes_worker, n_jobs=n_jobs, desc="Cluster sizes", verbose=verbose,
        ks=tuple(ks), linkages=tuple(linkages)))


# --------------------------------------------------------------------------- one traversal, three tables
def _sizes_rows(fits: Sequence["_PreparedFit"], ks: Sequence[int],
                linkages: Sequence[str]) -> List[Dict[str, float]]:
    """The partition-shape rows for one already-prepared group."""
    out = []
    for method in linkages:
        for k in sorted(set(ks) & set(fits[0].labels[method])):
            summaries = [cluster_size_summary(f.labels[method][k]) for f in fits]
            row = {"linkage": method, "k": int(k), "n_reps": len(fits)}
            for field in _SIZE_FIELDS:
                row.update(_mean_sem([s[field] for s in summaries], field))
            out.append(row)
    return out


def _dendrogram_rows(fits: Sequence["_PreparedFit"], linkages: Sequence[str],
                     pairs: Sequence[tuple]) -> List[Dict[str, float]]:
    """The k-free tree-agreement rows for one already-prepared group."""
    out = []
    for method in linkages:
        gammas = [baker_gamma(fits[i].coph_ranks[method], fits[j].coph_ranks[method])
                  for i, j in pairs]
        fidelity = [cophenetic_fidelity(f.trees[method], f.condensed) for f in fits]
        out.append({"linkage": method, "n_reps": len(fits), "n_pairs": len(gammas),
                    **_mean_sem(gammas, "baker_gamma"),
                    **_mean_sem(fidelity, "cophenetic_fidelity")})
    return out


def _agglomerative_worker(store_path, base, rows, ndim, ks, linkages, max_pairs, pair_seed):
    """All three agglomerative tables for one group, from ONE set of prepared fits."""
    from SpAM_Simulations.pipeline import _worker_store

    fits = _prepare_group(_worker_store(store_path), rows, ndim, ks, linkages)
    if not fits:
        return {"agreement": [], "dendrogram": [], "sizes": []}
    sizes = [{**base, **row} for row in _sizes_rows(fits, ks, linkages)]
    if len(fits) < 2:
        # Sizes are per-fit and still meaningful; the pair metrics have nothing to compare against.
        return {"agreement": [], "dendrogram": [], "sizes": sizes}
    pairs = sample_pairs(len(fits), max_pairs, pair_seed)
    return {
        "agreement": [{**base, **row}
                      for row in _agreement_rows(fits, ks, linkages, max_pairs, pair_seed)],
        "dendrogram": [{**base, **row} for row in _dendrogram_rows(fits, linkages, pairs)],
        "sizes": sizes,
    }


def compute_agglomerative_tables(store, ks: Sequence[int] = DEFAULT_KS,
                                 linkages: Sequence[str] = DEFAULT_LINKAGES,
                                 group_fields=None, verbose: bool = True, n_jobs: int = -1,
                                 max_pairs: Optional[int] = DEFAULT_MAX_PAIRS,
                                 pair_seed: int = 0) -> Dict[str, pd.DataFrame]:
    """``{"agreement", "dendrogram", "sizes"}`` from a SINGLE pass over the store.

    Identical output to calling :func:`compute_cluster_agreement`,
    :func:`compute_dendrogram_agreement` and :func:`compute_cluster_sizes` separately - and tests
    assert exactly that - but each cohort's linkage trees, cuts and cophenetic rankings are built
    once instead of three times. On a 17,729-fit store that redundancy was hours.

    The separate functions are kept: they are the right entry point when only one table is wanted,
    and they document each metric on its own terms.
    """
    from SpAM_Simulations.pipeline import group_tasks, map_groups_multi

    _require_conf(store)
    tasks = group_tasks(store, group_fields, min_size=1)
    merged = map_groups_multi(
        store, tasks, _agglomerative_worker, n_jobs=n_jobs, desc="Agglomerative tables",
        verbose=verbose, ks=tuple(ks), linkages=tuple(linkages), max_pairs=max_pairs,
        pair_seed=pair_seed)
    return {name: pd.DataFrame(merged.get(name, []))
            for name in ("agreement", "dendrogram", "sizes")}
