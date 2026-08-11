"""Density-based clustering as a **descriptive** companion to the agglomerative analysis.

`cluster_stability` answers "do two cohorts discover the same groups at granularity k?". It is the
right instrument for that, but it is structurally unable to answer a question stimulus construction
cares about just as much: **is this image confusable with anything at all?** Agglomerative
clustering assigns every one of the 725 images to some cluster, so an image that is genuinely
isolated is absorbed into whichever group happens to be nearest, and a deduplication rule read off
that partition will exclude it for no reason. At k=20 each cluster holds ~36 images, so this is not
a rare edge case.

HDBSCAN labels such points ``-1`` (noise), which is precisely the missing statement. An image
labelled noise by both cohorts is one nothing else is reliably confusable with, i.e. maximally safe
to use.

**A noise class is not a partition, but that does not make the labelling unusable for chaining.**
``-1`` is the absence of a cluster rather than a cluster, so treating it as one would make VI
dominated by a bucket that may hold most of the images, and the raw HDBSCAN labelling therefore
cannot carry `VI(cohort, paths) <= VI(cohort, cohort') + VI(cohort', paths)` as it stands.

The fix is to **restrict the ground set**. Drop the noise images from every labelling involved and
what remains are honest partitions of one shared subset, on which VI is a metric with all its usual
properties. The chained claim survives in a weaker, explicitly scoped form: *restricted to the
n-star images that all the labellings clustered*, the triangle inequality holds exactly, so an unmeasured
leg can still be bounded by the sum of two measured ones. :func:`common_clustered_mask` and
:func:`pairwise_restricted_vi` exist for precisely that, and they take **all** the labellings at
once, because scoring each pair on its own intersection puts the terms in different metric spaces
and they can no longer be added.

The price is stated rather than hidden: the surviving subset is chosen *by the clusterings*, so two
cohorts that both label aggressively keep only the easy, well-separated core and score well on it.
``vi_restricted`` is optimistically biased, the bias grows with the noise fraction, and every
function returning it also returns ``n_shared``/``frac_shared`` so the scope of any claim travels
with the number.

What is reported:

* **noise fraction and cluster count**, per cohort. HDBSCAN chooses the number of clusters rather
  than being told, so this is a genuinely independent read on the granularity question.
* **agreement on which images are noise**, which is a *binary* label per image, so plain Jaccard and
  Cohen's kappa apply directly and need no restriction at all.
* **VI and ARI over the jointly-clustered subset**, the semi-metric quantities above.

**Why not GMM anywhere.** At the granularities of interest there are as few as 3.6 images per
cluster (725 / 200), while a full covariance in 8-20 dimensions needs 36-210 parameters per
component, so full covariance is unfittable and the fallback is diagonal or spherical. Those are
**rotation-dependent**, and two cohorts' MDS solutions differ by an arbitrary rotation, so two
cohorts recovering geometrically identical spaces would be scored as disagreeing. Add that EM's
initialisation is stochastic, which would inject variance indistinguishable from cohort
disagreement into the very quantity being measured, and GMM is disqualified on both counts.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from tqdm.auto import tqdm

NOISE_LABEL = -1
# Swept in place of the agglomerative `k` grid. `min_cluster_size` is the smallest group HDBSCAN
# will call a cluster, so it sets granularity from below rather than by fiat; 2 is the loosest
# meaningful value (a confusable *pair* is exactly what stimulus construction must avoid).
DEFAULT_MIN_CLUSTER_SIZES = (2, 3, 5, 10, 20)


def hdbscan_labels(condensed: np.ndarray, min_cluster_size: int = 5,
                   min_samples: Optional[int] = None) -> np.ndarray:
    """Cluster one cohort's condensed distances. Returns labels with ``-1`` for noise.

    ``copy=True`` is passed explicitly rather than left to the default: sklearn's default is
    ``False``, which permits modifying the distance matrix in place, and callers here reuse a
    cached square matrix across several parameter settings.
    """
    if min_cluster_size < 2:
        raise ValueError(f"`min_cluster_size` must be at least 2, got {min_cluster_size}")
    square = squareform(np.asarray(condensed, dtype=np.float64))
    model = HDBSCAN(min_cluster_size=int(min_cluster_size),
                    min_samples=None if min_samples is None else int(min_samples),
                    metric="precomputed", copy=True)
    return model.fit(square).labels_


def noise_summary(labels: np.ndarray) -> Dict[str, float]:
    """Shape of one cohort's density partition, including how much of it is unclustered."""
    labels = np.asarray(labels)
    is_noise = labels == NOISE_LABEL
    clustered = labels[~is_noise]
    sizes = np.bincount(clustered) if clustered.size else np.array([], dtype=int)
    sizes = sizes[sizes > 0]
    return {
        "n_images": int(labels.size),
        "n_clusters": int(sizes.size),
        "frac_noise": float(is_noise.mean()),
        "n_clustered": int(clustered.size),
        "cluster_size_mean": float(sizes.mean()) if sizes.size else np.nan,
        "cluster_size_median": float(np.median(sizes)) if sizes.size else np.nan,
        "cluster_size_max": int(sizes.max()) if sizes.size else 0,
    }


def noise_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    """Do two cohorts agree on **which images are isolated**?

    The noise flag is a binary per-image label, so this needs no partition metric and none of the
    VI caveats apply. Three readings, because they fail differently:

    * ``noise_jaccard`` - overlap of the two noise *sets*. NaN when neither cohort flags anything,
      since the ratio is 0/0 and reporting 1.0 there would assert perfect agreement about nothing.
    * ``noise_kappa`` - Cohen's kappa on the binary flag, which is chance-corrected and therefore
      the honest headline when the noise fraction is far from 0.5.
    * ``both_noise_frac`` / ``either_noise_frac`` - the raw counts, so a high kappa driven by a
      handful of images cannot hide behind the coefficient.
    """
    a = np.asarray(labels_a) == NOISE_LABEL
    b = np.asarray(labels_b) == NOISE_LABEL
    if a.shape != b.shape:
        raise ValueError(f"label arrays must match in length, got {a.shape} and {b.shape}")
    both, either = int(np.sum(a & b)), int(np.sum(a | b))
    kappa = float(cohen_kappa_score(a, b)) if (a.any() or b.any()) and not (a.all() and b.all()) \
        else np.nan
    return {
        "noise_jaccard": both / either if either else np.nan,
        "noise_kappa": kappa,
        "both_noise_frac": both / a.size,
        "either_noise_frac": either / a.size,
        "frac_noise_a": float(a.mean()),
        "frac_noise_b": float(b.mean()),
    }


def common_clustered_mask(labellings: Sequence[np.ndarray]) -> np.ndarray:
    """Boolean mask of the images **every** supplied labelling assigned to some cluster.

    This is what makes a chained claim possible. Restricted to this mask each labelling is a genuine
    partition of the *same* ground set, so VI is a metric on it and the triangle inequality holds
    exactly. Pass all the labellings the chain will involve at once - two cohorts and a reference,
    say - rather than intersecting pairwise, because a pairwise-restricted VI is a distance in a
    different metric space for each pair and those cannot be added.
    """
    if not labellings:
        raise ValueError("need at least one labelling")
    mask = np.ones(len(labellings[0]), dtype=bool)
    for labels in labellings:
        labels = np.asarray(labels)
        if labels.shape != mask.shape:
            raise ValueError(f"labellings must match in length, got {labels.shape} and {mask.shape}")
        mask &= labels != NOISE_LABEL
    return mask


def restricted_vi(labels_a: np.ndarray, labels_b: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """VI between two density labellings, over the images both assigned to a cluster.

    Dropping the noise class from *both* sides leaves two honest partitions of the surviving subset,
    so on that subset VI is a metric with all its usual properties. That supports a real, if
    narrower, chained claim: **restricted to the images every labelling clustered**, one can still
    write ``VI(a, ref) <= VI(a, b) + VI(b, ref)``. Pass an explicit ``mask`` from
    :func:`common_clustered_mask` when the chain involves more than these two labellings, since
    otherwise each pair is scored on its own ground set and the terms are not addable.

    **The restriction is not neutral and the result is optimistically biased.** The surviving subset
    is chosen by the clusterings themselves, so two cohorts that both label aggressively keep only
    the easy, well-separated core and score well on it. The bias grows with the noise fraction,
    which is exactly why ``n_shared`` and ``frac_shared`` are returned alongside and why no
    comparison across settings is meaningful without them.

    ``vi_restricted_norm`` divides by ``log(n_shared)``, so it is interpretable within a pair but
    **not comparable across pairs with different subset sizes**; ``vi_restricted`` in nats is the
    quantity to add when chaining.
    """
    from SpAM_Simulations.measures.cluster_stability import variation_of_information

    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label arrays must match in length, got {a.shape} and {b.shape}")
    mask = common_clustered_mask([a, b]) if mask is None else np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    out = {"n_shared": n, "frac_shared": n / a.size if a.size else np.nan}
    # log(1) == 0 makes the normalised form undefined, and a single item carries no partition
    # structure to compare in any case.
    if n < 2:
        return {**out, "vi_restricted": np.nan, "vi_restricted_norm": np.nan}
    return {
        **out,
        "vi_restricted": variation_of_information(a[mask], b[mask], normalise=False),
        "vi_restricted_norm": variation_of_information(a[mask], b[mask], normalise=True),
    }


def pairwise_restricted_vi(labellings: Sequence[np.ndarray],
                           names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Every pairwise ``vi_restricted`` on the **single** ground set common to all ``labellings``.

    The form to use for a chained claim. Because one mask is shared by every pair, the returned
    values live in one metric space and the triangle inequality holds between any three of them, so
    an unmeasured leg can be bounded by the sum of two measured ones. Scoring each pair on its own
    intersection instead would silently mix metric spaces and the bound would not follow.

    ``n_shared`` is constant across rows by construction and is reported so the scope of the claim
    ("restricted to these n of 725 images") travels with the numbers.
    """
    from itertools import combinations

    labellings = [np.asarray(la) for la in labellings]
    names = list(names) if names is not None else [str(i) for i in range(len(labellings))]
    if len(names) != len(labellings):
        raise ValueError(f"got {len(names)} names for {len(labellings)} labellings")
    mask = common_clustered_mask(labellings)
    return pd.DataFrame([
        {"a": names[i], "b": names[j],
         **restricted_vi(labellings[i], labellings[j], mask=mask)}
        for i, j in combinations(range(len(labellings)), 2)
    ])


def clustered_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    """ARI over the images **both** cohorts assigned to some cluster. Descriptive only.

    Restricting to the jointly-clustered subset is what keeps this interpretable: including noise
    as though it were a cluster would let two cohorts that agree only on "most things are isolated"
    score as though they had agreed on a structure. The restriction also means the score is
    computed on a different subset for every parameter setting, so ``n_shared_clustered`` is
    reported alongside and a comparison across settings without it is meaningless.
    """
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    shared = (a != NOISE_LABEL) & (b != NOISE_LABEL)
    n = int(shared.sum())
    if n < 2 or len(set(a[shared])) < 1 or len(set(b[shared])) < 1:
        return {"ari_shared_clustered": np.nan, "n_shared_clustered": n,
                "frac_shared_clustered": n / a.size if a.size else np.nan}
    return {
        "ari_shared_clustered": float(adjusted_rand_score(a[shared], b[shared])),
        "n_shared_clustered": n,
        "frac_shared_clustered": n / a.size,
    }


def compare_density_partitions(condensed_a: np.ndarray, condensed_b: np.ndarray,
                               min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
                               min_samples: Optional[int] = None) -> pd.DataFrame:
    """One row per ``min_cluster_size``, comparing two cohorts. The reusable entry point."""
    rows = []
    for mcs in min_cluster_sizes:
        la = hdbscan_labels(condensed_a, mcs, min_samples)
        lb = hdbscan_labels(condensed_b, mcs, min_samples)
        sa, sb = noise_summary(la), noise_summary(lb)
        rows.append({
            "min_cluster_size": int(mcs),
            "n_clusters_a": sa["n_clusters"], "n_clusters_b": sb["n_clusters"],
            "mean_n_clusters": 0.5 * (sa["n_clusters"] + sb["n_clusters"]),
            "mean_frac_noise": 0.5 * (sa["frac_noise"] + sb["frac_noise"]),
            "mean_cluster_size": np.nanmean([sa["cluster_size_mean"], sb["cluster_size_mean"]]),
            **noise_agreement(la, lb), **clustered_ari(la, lb),
            **restricted_vi(la, lb),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- store driver
# Source field -> reported name. Per-cohort fields are averaged over reps, pair fields over the
# C(reps, 2) pairs; `cluster_size_mean` is renamed so the column is not `mean_cluster_size_mean`.
_MEAN_SEM_FIELDS = {
    "n_clusters": "n_clusters", "frac_noise": "frac_noise", "cluster_size_mean": "cluster_size",
    "noise_jaccard": "noise_jaccard", "noise_kappa": "noise_kappa",
    "both_noise_frac": "both_noise_frac", "either_noise_frac": "either_noise_frac",
    "ari_shared_clustered": "ari_shared_clustered",
    "frac_shared_clustered": "frac_shared_clustered",
    "vi_restricted": "vi_restricted", "vi_restricted_norm": "vi_restricted_norm",
}
# `restricted_vi` and `clustered_ari` both report the shared-subset size, identically whenever no
# explicit mask is passed. Only `frac_shared_clustered` is carried into the table, so the column is
# not duplicated; `n_shared` remains on the per-pair dict for standalone callers.


def _density_worker(store_path, base, rows, ndim, min_cluster_sizes, min_samples, max_pairs,
                    pair_seed):
    """One group's density agreement. Top-level so it can be pickled to a worker."""
    from SpAM_Simulations.measures.cluster_stability import _fit_distances, sample_pairs
    from SpAM_Simulations.core.pipeline import _worker_store

    store = _worker_store(store_path)
    dists = [_fit_distances(store, int(r), ndim) for r in rows]
    pairs = sample_pairs(len(dists), max_pairs, pair_seed)
    out = []
    for mcs in min_cluster_sizes:
        labels = [hdbscan_labels(d, mcs, min_samples) for d in dists]
        summaries = [noise_summary(la) for la in labels]
        # Each pair is restricted to its OWN jointly-clustered subset here, which is right for a
        # descriptive average but means these values are not addable across pairs. Use
        # `pairwise_restricted_vi` on one shared mask when chaining.
        scored = [{**noise_agreement(labels[i], labels[j]),
                   **clustered_ari(labels[i], labels[j]),
                   **restricted_vi(labels[i], labels[j])}
                  for i, j in pairs]
        row = {**base, "min_cluster_size": int(mcs), "n_reps": len(labels),
               "n_pairs": len(scored)}
        for field, name in _MEAN_SEM_FIELDS.items():
            vals = ([s[field] for s in summaries] if field in summaries[0]
                    else [p[field] for p in scored])
            row.update(_mean_sem(vals, name))
        out.append(row)
    return out


def compute_density_agreement(store, min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
                              min_samples: Optional[int] = None, group_fields=None,
                              verbose: bool = True, n_jobs: int = -1,
                              max_pairs: Optional[int] = None,
                              pair_seed: int = 0) -> pd.DataFrame:
    """Between-cohort density-clustering agreement per (configuration, ndim, min_cluster_size).

    Mirrors ``cluster_stability.compute_cluster_agreement``, including its caveat that the rep pairs
    are **not independent** (each cohort appears in many of them), so the reported SEM understates
    the true uncertainty.

    Labels are computed **once per fit** and reused across all pairs, since the pair loop is
    O(reps^2) while labelling is O(reps). That is also why ``max_pairs`` defaults to None here and
    not in the agglomerative pass: HDBSCAN labelling dominates this function, so sampling pairs
    saves little. Pass one to trade it anyway.
    """
    from SpAM_Simulations.measures.cluster_stability import _require_conf
    from SpAM_Simulations.core.pipeline import group_tasks, map_groups

    _require_conf(store)
    tasks = group_tasks(store, group_fields, min_size=2)
    return pd.DataFrame(map_groups(
        store, tasks, _density_worker, n_jobs=n_jobs, desc="Density agreement", verbose=verbose,
        min_cluster_sizes=tuple(min_cluster_sizes), min_samples=min_samples,
        max_pairs=max_pairs, pair_seed=pair_seed))


def _mean_sem(values: Sequence[float], name: str) -> Dict[str, float]:
    """Mean and SEM ignoring NaNs, which are legitimate here (see :func:`noise_agreement`)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"mean_{name}": np.nan, f"sem_{name}": np.nan}
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {f"mean_{name}": float(arr.mean()), f"sem_{name}": sem}


def _isolated_worker(store_path, base, rows, ndim, min_cluster_size, min_samples):
    """One group's per-image isolation fractions. Top-level so it can be pickled to a worker."""
    from SpAM_Simulations.measures.cluster_stability import _fit_distances
    from SpAM_Simulations.core.pipeline import _worker_store

    store = _worker_store(store_path)
    flags = [hdbscan_labels(_fit_distances(store, int(r), ndim), min_cluster_size,
                            min_samples) == NOISE_LABEL for r in rows]
    if not flags:
        return []
    frac = np.mean(np.vstack(flags), axis=0)
    return [{**base, "min_cluster_size": int(min_cluster_size), "image": int(i),
             "frac_cohorts_noise": float(f), "n_reps": len(flags)}
            for i, f in enumerate(frac)]


def isolated_images(store, min_cluster_size: int = 5, min_samples: Optional[int] = None,
                    group_fields=None, verbose: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    """Per-image: in what fraction of this group's cohorts was this image unclustered?

    The directly actionable table. An image with ``frac_cohorts_noise`` near 1 is one that no
    cohort found reliably confusable with anything, so it is the safest kind of stimulus to use;
    an image near 0 always sits inside some group and its group-mates are the exclusion set.
    """
    from SpAM_Simulations.measures.cluster_stability import _require_conf
    from SpAM_Simulations.core.pipeline import group_tasks, map_groups

    _require_conf(store)
    tasks = group_tasks(store, group_fields, min_size=1)
    return pd.DataFrame(map_groups(
        store, tasks, _isolated_worker, n_jobs=n_jobs, desc="Isolated images", verbose=verbose,
        min_cluster_size=min_cluster_size, min_samples=min_samples))
