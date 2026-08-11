"""Task-agnostic construction of a ground-truth embedding from empirical SpAM data.

Takes any collection of subjects who judged pairs on a shared image index and returns coordinates
plus the evidence for the dimensionality chosen. Nothing here knows which task version produced the
data, so it serves v1 through v4 alike.

**Why this module exists.** Dimensionality used to be picked by reading a classical-MDS
eigenspectrum off the *mean-imputed* aggregate RDM and taking the smallest number of dimensions
explaining 90% of variance, capped at 15. That is invalid on this data. 63.6% of pairs are
unobserved and were filled with a single constant, which asserts that all those point pairs are
equidistant; k mutually equidistant points form a regular simplex needing k-1 dimensions, so the
fill manufactures rank rather than merely adding noise. Measured: a synthetic **rank-8** space put
through the identical mask and fill reports an effective rank of 193 and needs 239 dimensions for
90% variance, statistically indistinguishable from the real data's 213 and 216. The rule therefore
returned its cap, 15, carrying almost no information about the data, and the resulting embedding was
near-isotropic.

**What replaces it.** Dimensionality is a generalisation question, not a linear-algebra one, so it
is answered by out-of-sample prediction and no imputation is used anywhere:

* :func:`dimensionality_scan` fits each candidate dimensionality on **disjoint halves of the
  subjects** and scores how well the two halves agree. Weighted SMACOF treats weight 0 as missing,
  so only observed pairs ever enter a fit.
* :func:`cross_validate_ndim` verifies the choice by leave-k-out over subjects: fit on the rest,
  then predict the held-out subjects' own observed distances.

Both are reported at global (Spearman over reconstructed distance vectors, Procrustes M^2 over
configurations) and local (top-5% closest-pair Jaccard) resolution, but :func:`select_ndim` selects
on the **global** criterion by default.

**Connectivity is the binding constraint.** ``run_mds`` refuses a disconnected pair graph, and at
this coverage a random half of the pilot is connected only ~90% of the time. :func:`draw_valid_splits`
discards and redraws such splits, and reports the discard rate together with the coverage of
discarded versus kept draws, because discarding is not neutral: it preferentially keeps splits
containing well-covered subjects.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from SpAM_Simulations.measures.metrics import topk_similar_jaccard

DEFAULT_NDIMS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20)
DEFAULT_TOP_FRAC = 0.05


# --------------------------------------------------------------------------- aggregation
def aggregate_subjects(subjects: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Pool subjects into ``(mean_distances, weights)``, **without** a connectivity check.

    ``mean_distances`` is the observation-count-weighted per-pair mean (0 where unobserved) and
    ``weights`` the matching 0/1 mask, which is exactly ``run_mds``'s input contract. Callers that
    require a connected graph should use :func:`is_connected` or ``pilot.pilot_aggregate``; this
    function stays silent so the split search can *test* candidate subsets cheaply.
    """
    if not subjects:
        raise ValueError("no subjects to aggregate")
    n_pairs = subjects[0].distances.shape[0]
    total = np.zeros(n_pairs, dtype=np.float64)
    count = np.zeros(n_pairs, dtype=np.int64)
    for s in subjects:
        obs = s.n_obs > 0
        total[obs] += np.nan_to_num(s.distances[obs]) * s.n_obs[obs]
        count += s.n_obs
    weights = (count > 0).astype(np.float32)
    mean_distances = np.where(count > 0, total / np.maximum(count, 1), 0.0).astype(np.float32)
    return mean_distances, weights


def observed_mask(subjects: Sequence) -> np.ndarray:
    """0/1 condensed mask of which pairs anyone in ``subjects`` judged."""
    return aggregate_subjects(subjects)[1]


def n_components(weights: np.ndarray) -> int:
    """Connected components of the observed-pair graph."""
    return int(connected_components(squareform(weights), directed=False, return_labels=False))


def is_connected(subjects: Sequence) -> bool:
    """Whether ``subjects`` jointly observe a connected pair graph, i.e. whether MDS can run."""
    return n_components(observed_mask(subjects)) == 1


def coverage_of(subjects: Sequence) -> float:
    """Fraction of all pairs observed at least once."""
    return float(observed_mask(subjects).mean())


# --------------------------------------------------------------------------- split-half search
def draw_valid_splits(subjects: Sequence, n_draws: int, rng: np.random.Generator,
                      ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """``n_draws`` strictly disjoint half-splits whose halves are both connected.

    Returns ``(splits, diagnostics)`` where each split is a pair of index arrays into ``subjects``.
    Splits whose either half is disconnected are discarded and redrawn until ``n_draws`` usable ones
    exist.

    The diagnostics matter as much as the splits. Discarding *could* be a biased filter: if a half
    is disconnected precisely when it holds poorly-covered subjects, the kept draws over-represent
    well-covered ones and the agreement estimates are optimistic.

    **Read ``coverage_gap_frac``, not ``discard_rate``.** The rate alone says only that the subject
    pool is sparse; it is the coverage *gap* between kept and discarded draws that says whether the
    filter is selective. On the pilot the two come apart sharply: a 38-42% discard rate with a gap of
    +0.0007 on a base of 0.166, i.e. **0.4% relative** - sparse, but not measurably biased.

    A high rate with a negligible gap is expected here and is a property of the pilot's composition
    rather than of the design being planned. Of the 41 pre-SHINE subjects, 25 ran task v1/v2 with
    only 10 trials each (1,873 observed pairs) against v3's 16 subjects at 3,230 - so a half drawn
    heavy on v1/v2 is disproportionately likely to be disconnected. The deployed v4 session collects
    18 distinct trials, so this is a pilot artifact, not a forecast for the study.
    """
    if n_draws <= 0:
        raise ValueError(f"n_draws must be positive, got {n_draws}")
    n = len(subjects)
    if n < 4:
        raise ValueError(f"need at least 4 subjects to split, got {n}")
    half = n // 2

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    n_attempts = 0
    cov_kept: List[float] = []
    cov_discarded: List[float] = []
    max_attempts = max(200, n_draws * 50)
    while len(splits) < n_draws:
        if n_attempts >= max_attempts:
            raise RuntimeError(
                f"only {len(splits)} of {n_draws} splits were connected after {n_attempts} "
                f"attempts ({len(splits) / max(n_attempts, 1):.1%} usable). The subject pool is too "
                f"sparse to split; use leave-k-out cross-validation alone, or restrict the image set."
            )
        n_attempts += 1
        perm = rng.permutation(n)
        a, b = perm[:half], perm[half:]
        sa = [subjects[i] for i in a]
        sb = [subjects[i] for i in b]
        # The BINDING half, not the average of the two. `sa` and `sb` are complementary, so a
        # well-covered half forces a poorly-covered partner and `0.5 * (cov_a + cov_b)` is very
        # nearly invariant across permutations - measured on the pilot it has sd 0.0001 and
        # reported 0.172 for kept and discarded alike, i.e. it could not detect a gap of any size.
        # A split is discarded because ONE half is disconnected, and that is the poorly-covered
        # one, so the minimum is the quantity with any power to show the filter's selectivity
        # (sd 0.0039, ~40x more variance).
        cov = min(coverage_of(sa), coverage_of(sb))
        if is_connected(sa) and is_connected(sb):
            splits.append((a, b))
            cov_kept.append(cov)
        else:
            cov_discarded.append(cov)

    kept_mean = float(np.mean(cov_kept)) if cov_kept else np.nan
    disc_mean = float(np.mean(cov_discarded)) if cov_discarded else np.nan
    gap = kept_mean - disc_mean
    return splits, {
        "n_draws": len(splits),
        "n_attempts": n_attempts,
        "n_discarded": len(cov_discarded),
        "discard_rate": len(cov_discarded) / n_attempts,
        "half_size": half,
        # Binding-half coverage: the minimum of the two halves. See the comment above for why the
        # average of the two is not a usable statistic here.
        "mean_binding_coverage_kept": kept_mean,
        "mean_binding_coverage_discarded": disc_mean,
        "coverage_gap": gap,
        # The gap as a fraction of the kept level, which is what a threshold should read: an
        # absolute gap of 0.001 means something very different at 0.17 coverage than at 0.9.
        "coverage_gap_frac": gap / kept_mean if kept_mean else np.nan,
    }


# --------------------------------------------------------------------------- embedding
def _classical_embed(condensed: np.ndarray, ndim: int) -> np.ndarray:
    """Classical-MDS (PCoA) coordinates: double-centre the squared distances, keep top-``ndim``."""
    sq = squareform(condensed).astype(np.float64) ** 2
    n = sq.shape[0]
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ sq @ centring
    vals, vecs = np.linalg.eigh(gram)
    idx = np.argsort(vals)[::-1][:ndim]
    return (vecs[:, idx] * np.sqrt(np.clip(vals[idx], 0, None))).astype(np.float32)


def embed_subset(subjects: Sequence, ndim: int, method: str = "smacof",
                 max_iters: int = 1000, convergence_tol: float = 1e-6,
                 precalc_init: bool = True) -> np.ndarray:
    """Embed one subject subset in ``ndim`` dimensions. Returns ``(n_images, ndim)`` float32.

    ``method="smacof"`` is the canonical path: weighted SMACOF via ``run_mds``, which honours the
    0/1 weights so unobserved pairs never enter the fit. Needs R + rpy2 + smacof.

    ``method="classical"`` is a **smoke-test path only**. Classical MDS needs a complete matrix, so
    it mean-imputes the unobserved pairs, which is precisely the operation this module exists to
    avoid: on sparse data it manufactures dimensionality. Use it to exercise plumbing without R,
    never to select a dimensionality.
    """
    dists, weights = aggregate_subjects(subjects)
    if method == "classical":
        imputed = dists.copy()
        observed = weights > 0
        if not observed.all():
            imputed[~observed] = dists[observed].mean()
        return _classical_embed(imputed, ndim)
    if method != "smacof":
        raise ValueError(f"method must be 'smacof' or 'classical', got {method!r}")
    from SpAM_Simulations.core.multi_dimensional_scaling import run_mds  # lazy: imports R
    out = run_mds(dists=dists, weights=weights, ndim=ndim, max_iters=max_iters,
                  convergence_tol=convergence_tol, precalc_init=precalc_init)
    return np.asarray(out["conf"], dtype=np.float32)


# --------------------------------------------------------------------------- scoring
def split_half_scores(coords_a: np.ndarray, coords_b: np.ndarray,
                      top_frac: float = DEFAULT_TOP_FRAC) -> Dict[str, float]:
    """Agreement between two independently-fitted configurations of the same images.

    * ``spearman`` - rank agreement of the reconstructed distance vectors; higher is better.
    * ``procrustes_m2`` - Procrustes M^2 after optimal translation/scale/rotation, i.e. what remains
      once MDS's gauge freedom is removed; **lower** is better.
    * ``topk_jaccard`` - overlap of the closest-``top_frac`` pair sets, the decision-relevant
      quantity when the goal is flagging items that are too similar; higher is better.
    """
    da, db = pdist(coords_a), pdist(coords_b)
    _, _, m2 = procrustes(coords_a, coords_b)
    return {
        "spearman": float(spearmanr(da, db).statistic),
        "procrustes_m2": float(m2),
        "topk_jaccard": float(topk_similar_jaccard(da, db, top_frac)),
    }


# --------------------------------------------------------------------------- the scan
def dimensionality_scan(subjects: Sequence, ndims: Sequence[int] = DEFAULT_NDIMS,
                        n_draws: int = 50, seed: int = 0, method: str = "smacof",
                        top_frac: float = DEFAULT_TOP_FRAC, verbose: bool = True,
                        splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                        ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Score every candidate dimensionality by split-half agreement.

    Returns ``(long_dataframe, split_diagnostics)`` with one row per (ndim, draw) carrying all three
    scores, so the distribution is inspectable rather than pre-summarised.

    **The same splits are reused across every ndim.** Drawing fresh splits per dimensionality would
    cost the same but leave the comparison unpaired, and since the curve is expected to be nearly
    flat, between-draw variance would swamp the between-ndim differences the scan exists to measure.
    """
    ndims = [int(d) for d in ndims]
    if not ndims:
        raise ValueError("`ndims` must be non-empty")
    rng = np.random.default_rng(seed)
    diagnostics: Dict[str, float] = {}
    if splits is None:
        splits, diagnostics = draw_valid_splits(subjects, n_draws, rng)

    rows = []
    total = len(ndims) * len(splits)
    with tqdm(total=total, desc="Dimensionality scan", disable=not verbose) as bar:
        for ndim in ndims:
            for draw, (ia, ib) in enumerate(splits):
                ca = embed_subset([subjects[i] for i in ia], ndim, method=method)
                cb = embed_subset([subjects[i] for i in ib], ndim, method=method)
                rows.append({"ndim": ndim, "draw": draw, **split_half_scores(ca, cb, top_frac)})
                bar.update(1)
    return pd.DataFrame(rows), diagnostics


# --------------------------------------------------------------------------- parallel execution
# The serial scan above is the readable reference and is what the tests exercise. At the sizes the
# EC2 run needs (11 ndims x 50 draws x 2 halves = 1100 fits, plus 11 x 40 CV folds) it is far too
# slow, so the same work is also available as picklable payloads for joblib. Both paths call the
# same `run_mds` and score with the same `split_half_scores`.
def split_aggregates(subjects: Sequence, splits: Sequence[Tuple[np.ndarray, np.ndarray]],
                     ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Pre-pool each half of each split into ``(distances, weights)``, once for all dimensionalities.

    Aggregation does not depend on ``ndim``, and a worker payload must carry arrays rather than
    subject objects: pooling here keeps the payload at two condensed vectors (~2 MB) instead of
    shipping the whole subject list to every task.
    """
    return [(aggregate_subjects([subjects[i] for i in ia]),
             aggregate_subjects([subjects[i] for i in ib])) for ia, ib in splits]


def _embed_payload(payload: tuple) -> tuple:
    """Fit one weighted-SMACOF embedding. Module-level and picklable, so joblib can ship it.

    Mirrors ``pipeline._execute_mds_payload``: ``run_mds`` (and therefore R) is imported inside the
    worker so each process initialises its own R exactly once, and a failure is *recorded* rather
    than raised, because one disconnected draw must not abort a multi-hour scan.
    """
    from SpAM_Simulations.core.multi_dimensional_scaling import run_mds
    key, dists, weights, ndim, max_iters, tol, precalc = payload
    try:
        out = run_mds(dists=dists, weights=weights, ndim=ndim, max_iters=max_iters,
                      convergence_tol=tol, precalc_init=precalc)
    except RuntimeError as exc:
        status = "disconnected" if "connected components" in str(exc) else "error"
        return key, None, {"status": status, "niter": np.nan, "stress": np.nan}
    return (key, np.asarray(out["conf"], dtype=np.float32).reshape(-1, ndim),
            {"status": "max_iters" if out["needs_more_iters"] else "success",
             "niter": float(out["niter"]), "stress": float(out["stress"])})


def _run_embeddings(payloads, total: int, n_jobs: int = 1, verbose: bool = True,
                    desc: str = "Embedding") -> Dict:
    """Execute embedding payloads, serially or across ``n_jobs`` processes, keyed by their ``key``.

    ``payloads`` should be a generator: each one holds full-length distance and weight vectors, so
    materialising a whole scan's worth at once costs gigabytes for no benefit.
    """
    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_embed_payload)(p) for p in payloads
        )
    else:
        results = (_embed_payload(p) for p in payloads)
    out: Dict = {}
    with tqdm(total=total, desc=desc, disable=not verbose) as bar:
        for key, coords, info in results:
            out[key] = (coords, info)
            bar.update(1)
    return out


def scan_ndim_parallel(aggregates: Sequence, ndim: int, n_jobs: int = 1,
                       top_frac: float = DEFAULT_TOP_FRAC, max_iters: int = 1000,
                       convergence_tol: float = 1e-6, precalc_init: bool = True,
                       verbose: bool = True) -> pd.DataFrame:
    """Split-half scores at **one** dimensionality, over pre-pooled ``split_aggregates``.

    One ndim per call rather than the whole grid, so a long EC2 loop can checkpoint its partial
    results to S3 after each dimensionality instead of only at the end.

    Rows carry ``status_a``/``status_b`` and ``niter_a``/``niter_b`` alongside the scores. Those are
    not decoration: a half-sized subject pool hits ``max_iters`` far more often than the dense
    aggregate does, and a scan where most fits never converged is measuring the solver's stopping
    rule rather than the data's dimensionality.
    """
    payloads = ((("a" if half == 0 else "b", draw), agg[0], agg[1], int(ndim),
                 max_iters, convergence_tol, precalc_init)
                for draw, pair in enumerate(aggregates)
                for half, agg in enumerate(pair))
    fits = _run_embeddings(payloads, total=2 * len(aggregates), n_jobs=n_jobs, verbose=verbose,
                           desc=f"Split-half ndim={ndim}")
    rows = []
    for draw in range(len(aggregates)):
        (ca, ia), (cb, ib) = fits[("a", draw)], fits[("b", draw)]
        row = {"ndim": int(ndim), "draw": draw,
               "status_a": ia["status"], "status_b": ib["status"],
               "niter_a": ia["niter"], "niter_b": ib["niter"],
               "stress_a": ia["stress"], "stress_b": ib["stress"]}
        scores = (split_half_scores(ca, cb, top_frac) if ca is not None and cb is not None
                  else {"spearman": np.nan, "procrustes_m2": np.nan, "topk_jaccard": np.nan})
        rows.append({**row, **scores})
    return pd.DataFrame(rows).sort_values("draw").reset_index(drop=True)


def cross_validate_ndim_parallel(subjects: Sequence, ndim: int,
                                 folds: Sequence[np.ndarray], n_jobs: int = 1,
                                 max_iters: int = 1000, convergence_tol: float = 1e-6,
                                 precalc_init: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Leave-k-out CV at **one** dimensionality, parallel over folds.

    Same question as :func:`cross_validate_ndim` - generalisation to unseen *people* - and the folds
    are passed in so every dimensionality is scored on identical hold-outs, keeping the comparison
    paired.
    """
    n = len(subjects)
    train_sets = []
    for held in folds:
        mask = np.ones(n, dtype=bool)
        mask[np.asarray(held)] = False
        train_sets.append(np.flatnonzero(mask))
    payloads = ((fold, *aggregate_subjects([subjects[i] for i in train]), int(ndim),
                 max_iters, convergence_tol, precalc_init)
                for fold, train in enumerate(train_sets))
    fits = _run_embeddings(payloads, total=len(folds), n_jobs=n_jobs, verbose=verbose,
                           desc=f"Leave-k-out ndim={ndim}")
    rows = []
    for fold, held in enumerate(folds):
        coords, info = fits[fold]
        score = (_score_held_out(coords, [subjects[i] for i in held])
                 if coords is not None else np.nan)
        rows.append({"ndim": int(ndim), "fold": fold, "spearman": score,
                     "status": info["status"], "niter": info["niter"]})
    return pd.DataFrame(rows)


DEFAULT_SCAN_METRICS = ("spearman", "procrustes_m2", "topk_jaccard")


def summarise_scan(scan: pd.DataFrame, by: str = "ndim",
                   metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Mean and standard error of each score per level of ``by``.

    ``by``/``metrics`` are parameters rather than constants so the same summary serves the
    dimensionality scan and the cluster-agreement sweep, whose long frame is keyed on ``k`` and
    carries different columns.
    """
    cols = list(metrics) if metrics is not None else [m for m in DEFAULT_SCAN_METRICS
                                                      if m in scan.columns]
    g = scan.groupby(by)[cols]
    out = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_sem"))
    return out.reset_index().sort_values(by).reset_index(drop=True)


# Direction of each metric. Everything here is higher-is-better except the two that measure residual
# disagreement: Procrustes M^2 and Variation of Information.
_HIGHER_IS_BETTER = {
    "spearman": True, "procrustes_m2": False, "topk_jaccard": True,
    "vi": False, "vi_norm": False, "ari": True, "ami": True,
    "sil_cross": True, "sil_ratio": True, "jaccard_mean": True, "baker_gamma": True,
}


def select_ndim(scan: pd.DataFrame, criterion: str = "spearman", rule: str = "one_se",
                axis: str = "ndim") -> int:
    """Choose a dimensionality from a scan.

    ``axis`` names the swept column, so the same rule selects a cluster granularity from a
    ``(k, metric...)`` frame; ``cluster_stability.select_k`` is a thin wrapper that does exactly that.

    ``rule="one_se"`` takes the **smallest ndim whose mean is within one standard error of the best
    mean**, the standard cross-validation heuristic (Breiman; ``glmnet``'s ``lambda.1se``). Two
    reasons it suits this problem: the best score is itself noisy, so candidates within 1 SE are not
    distinguishable and parsimony should break the tie; and on a nearly flat curve a plain argmax is
    driven by noise and drifts toward high ndim. That drift is the observed failure mode - in the
    existing sweep, Procrustes M^2 degraded from 0.428 at ndim 5 to 0.532 at ndim 10 while Spearman
    stayed flat at 0.78, i.e. more dimensions bought nothing and generalised worse.

    ``rule="argmax"`` takes the plain optimum, for comparison only.
    """
    if criterion not in _HIGHER_IS_BETTER:
        raise ValueError(f"criterion must be one of {sorted(_HIGHER_IS_BETTER)}, got {criterion!r}")
    summary = summarise_scan(scan, by=axis, metrics=[criterion])
    return apply_selection_rule(
        summary[axis].to_numpy(), summary[f"{criterion}_mean"].to_numpy(),
        summary[f"{criterion}_sem"].to_numpy(), higher_is_better=_HIGHER_IS_BETTER[criterion],
        rule=rule,
    )


def apply_selection_rule(levels: np.ndarray, means: np.ndarray, sems: np.ndarray,
                         higher_is_better: bool, rule: str = "one_se") -> int:
    """The selection rule itself, over already-summarised ``(level, mean, sem)`` triples.

    Factored out so :func:`select_ndim` and ``cluster_stability.select_k`` share one implementation
    despite starting from differently-shaped frames: the dimensionality scan holds one row per draw
    and must be summarised first, whereas the cluster sweep already stores ``mean_``/``sem_``
    columns. Passing the latter through a re-summarising path would compute a SEM over a single
    value, yield NaN, and silently degrade the one-SE rule to a plain argmax.

    ``levels`` must be sorted ascending, so "first qualifying" is "smallest qualifying".
    """
    means = np.asarray(means, dtype=float)
    sems = np.asarray(sems, dtype=float)
    best_i = int(np.argmax(means) if higher_is_better else np.argmin(means))
    if rule == "argmax":
        return int(levels[best_i])
    if rule != "one_se":
        raise ValueError(f"rule must be 'one_se' or 'argmax', got {rule!r}")
    se = sems[best_i]
    if not np.isfinite(se):
        return int(levels[best_i])
    threshold = means[best_i] - se if higher_is_better else means[best_i] + se
    ok = means >= threshold if higher_is_better else means <= threshold
    return int(levels[np.flatnonzero(ok)[0]])


# --------------------------------------------------------------------------- cross-validation
def leave_k_out_folds(n_subjects: int, k: int = 5, n_folds: int = 40,
                      rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """``n_folds`` random hold-out sets of ``k`` subject indices (sampling without replacement)."""
    if not 0 < k < n_subjects:
        raise ValueError(f"k must be in (0, n_subjects), got k={k}, n_subjects={n_subjects}")
    rng = rng if rng is not None else np.random.default_rng(0)
    return [rng.choice(n_subjects, size=k, replace=False) for _ in range(n_folds)]


def _score_held_out(coords: np.ndarray, held_out: Sequence) -> float:
    """Mean Spearman of each held-out subject's observed distances against the fitted geometry.

    Scored per subject and then averaged, rather than pooling every held-out observation, so a
    subject who happens to have judged more pairs does not dominate the fold.
    """
    implied = pdist(coords)
    scores = []
    for s in held_out:
        obs = np.flatnonzero(s.n_obs > 0)
        if obs.size < 2:
            continue
        r = spearmanr(s.distances[obs], implied[obs]).statistic
        if np.isfinite(r):
            scores.append(float(r))
    return float(np.mean(scores)) if scores else np.nan


def cross_validate_ndim(subjects: Sequence, ndims: Sequence[int] = DEFAULT_NDIMS,
                        k: int = 5, n_folds: int = 40, seed: int = 0, method: str = "smacof",
                        verbose: bool = True) -> pd.DataFrame:
    """Leave-k-out over **subjects**: fit on the rest, predict the held-out subjects' own distances.

    Returns one row per (ndim, fold). This verifies the split-half choice against a different
    question - generalisation to unseen *people* rather than agreement between two halves - so the
    two curves agreeing is meaningful corroboration and their disagreeing is a finding.

    Holding out subjects rather than pairs keeps every fit dense enough to stay connected: at k=5 a
    fit still sees n-5 subjects, comfortably above the ~25 needed for a connected pilot graph.
    """
    ndims = [int(d) for d in ndims]
    n = len(subjects)
    folds = leave_k_out_folds(n, k=k, n_folds=n_folds, rng=np.random.default_rng(seed))
    rows = []
    with tqdm(total=len(ndims) * len(folds), desc="Leave-k-out CV", disable=not verbose) as bar:
        for ndim in ndims:
            for fold, held in enumerate(folds):
                mask = np.ones(n, dtype=bool)
                mask[held] = False
                train = [subjects[i] for i in np.flatnonzero(mask)]
                coords = embed_subset(train, ndim, method=method)
                rows.append({"ndim": ndim, "fold": fold,
                             "spearman": _score_held_out(coords, [subjects[i] for i in held])})
                bar.update(1)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- final embedding
def build_gt(subjects: Sequence, ndim: int, method: str = "smacof") -> Tuple[np.ndarray, dict]:
    """Fit the final ground-truth embedding on **all** ``subjects`` at the chosen ``ndim``.

    ``ndim`` is required: there is no defensible way to infer it from a single fit, which is exactly
    the mistake the retired ``_choose_n_dims`` made. Get it from :func:`select_ndim`.
    """
    if ndim <= 0:
        raise ValueError(f"`ndim` must be positive, got {ndim}")
    weights = observed_mask(subjects)
    comps = n_components(weights)
    if comps > 1:
        raise RuntimeError(
            f"observed-pair graph has {comps} connected components (only {weights.mean():.1%} of "
            f"pairs observed); refusing to fit a ground truth on a partial graph"
        )
    coords = embed_subset(subjects, ndim, method=method)
    return coords, {
        "n_dims": int(ndim),
        "method": method,
        "n_subjects": len(subjects),
        "observed_frac": float(weights.mean()),
        "variants": sorted({getattr(s, "shine_variant", "") for s in subjects}),
        "task_versions": sorted({float(s.task_version) for s in subjects}),
    }
