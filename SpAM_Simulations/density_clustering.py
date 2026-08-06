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

**This module is deliberately secondary, and its outputs must not enter the transitivity chain.**
Variation of Information is primary in `cluster_stability` because it is a true metric on
*partitions*, which is what licenses `VI(cohort, paths) <= VI(cohort, cohort') + VI(cohort', paths)`
in the later path-hierarchy comparison. A labelling with a noise class is not a partition in that
sense: ``-1`` is not a cluster, it is the absence of one, and treating it as a cluster would make VI
dominated by a bucket that may hold most of the images. So nothing here is reported as VI, and
nothing here may be substituted into that argument.

What is reported instead needs no metricity:

* **noise fraction and cluster count**, per cohort. HDBSCAN chooses the number of clusters rather
  than being told, so this is a genuinely independent read on the granularity question.
* **agreement on which images are noise**, which is a *binary* label per image, so plain Jaccard and
  Cohen's kappa apply directly.
* **ARI restricted to the images both cohorts clustered**, as a descriptive sanity check only.

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
}


def compute_density_agreement(store, min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
                              min_samples: Optional[int] = None, group_fields=None,
                              verbose: bool = True) -> pd.DataFrame:
    """Between-cohort density-clustering agreement per (configuration, ndim, min_cluster_size).

    Mirrors ``cluster_stability.compute_cluster_agreement``, including its caveat that the
    C(n_reps, 2) rep pairs are **not independent** (each cohort appears in n_reps - 1 of them), so
    the reported SEM understates the true uncertainty.

    Labels are computed **once per fit** and reused across all pairs, since the pair loop is
    O(reps²) while labelling is O(reps).
    """
    from itertools import combinations

    from SpAM_Simulations.cluster_stability import _fit_distances, _require_conf
    from SpAM_Simulations.pipeline import _grouped_successful

    _require_conf(store)
    grouped, group_fields = _grouped_successful(store, group_fields)
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Density agreement",
                         disable=not verbose):
        if len(grp) < 2:
            continue          # a lone successful rep has nothing to be compared against
        ndim = int(grp["ndim"].iloc[0])
        dists = [_fit_distances(store, int(r), ndim) for r in grp["confdist_row"]]
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_fields, key_tuple))
        for mcs in min_cluster_sizes:
            labels = [hdbscan_labels(d, mcs, min_samples) for d in dists]
            summaries = [noise_summary(la) for la in labels]
            pairs = [{**noise_agreement(labels[i], labels[j]), **clustered_ari(labels[i], labels[j])}
                     for i, j in combinations(range(len(labels)), 2)]
            row = {**base, "min_cluster_size": int(mcs), "n_reps": len(labels),
                   "n_pairs": len(pairs)}
            for field, name in _MEAN_SEM_FIELDS.items():
                vals = ([s[field] for s in summaries] if field in summaries[0]
                        else [p[field] for p in pairs])
                row.update(_mean_sem(vals, name))
            rows.append(row)
    return pd.DataFrame(rows)


def _mean_sem(values: Sequence[float], name: str) -> Dict[str, float]:
    """Mean and SEM ignoring NaNs, which are legitimate here (see :func:`noise_agreement`)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"mean_{name}": np.nan, f"sem_{name}": np.nan}
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {f"mean_{name}": float(arr.mean()), f"sem_{name}": sem}


def isolated_images(store, min_cluster_size: int = 5, min_samples: Optional[int] = None,
                    group_fields=None, verbose: bool = True) -> pd.DataFrame:
    """Per-image: in what fraction of this group's cohorts was this image unclustered?

    The directly actionable table. An image with ``frac_cohorts_noise`` near 1 is one that no
    cohort found reliably confusable with anything, so it is the safest kind of stimulus to use;
    an image near 0 always sits inside some group and its group-mates are the exclusion set.
    """
    from SpAM_Simulations.cluster_stability import _fit_distances, _require_conf
    from SpAM_Simulations.pipeline import _grouped_successful

    _require_conf(store)
    grouped, group_fields = _grouped_successful(store, group_fields)
    rows = []
    for key, grp in tqdm(grouped, total=grouped.ngroups, desc="Isolated images",
                         disable=not verbose):
        ndim = int(grp["ndim"].iloc[0])
        flags = []
        for r in grp["confdist_row"]:
            labels = hdbscan_labels(_fit_distances(store, int(r), ndim), min_cluster_size,
                                    min_samples)
            flags.append(labels == NOISE_LABEL)
        if not flags:
            continue
        frac = np.mean(np.vstack(flags), axis=0)
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_fields, key_tuple))
        rows.extend({**base, "min_cluster_size": int(min_cluster_size), "image": int(i),
                     "frac_cohorts_noise": float(f), "n_reps": len(flags)}
                    for i, f in enumerate(frac))
    return pd.DataFrame(rows)
