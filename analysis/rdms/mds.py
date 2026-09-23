"""
MDS-embedding logic layered on top of SpAM_Simulations.core.multi_dimensional_scaling.run_mds():
classical (Torgerson) eigen-analysis of a raw RDM, a cached weighted-SMACOF sweep across
dimensionalities, and comparison of two fitted embeddings (Procrustes m^2, confdist
correlation, verdict classification). run_mds() itself and its input-conversion primitives
(condensed_rdm_to_weighted_dists, procrustes_m2, double_centered_gram) stay in
SpAM_Simulations/core/ -- this module is the analysis-side layer that caches sweeps to disk
and compares results across cohorts/conditions, not a second copy of the fitting logic.

Usage (from repo root):
    from analysis.rdms.mds import (
        classical_mds_eigenspectrum, effective_rank, run_mds_sweep,
        compare_mds_configs, classify_mds_agreement,
    )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from SpAM_Simulations.core.helpers import condensed_rdm_to_weighted_dists, double_centered_gram, procrustes_m2

# ---------------------------------------------------------------------------
# Classical (Torgerson) eigenspectrum -- qualitative only, see caveat below
# ---------------------------------------------------------------------------


def classical_mds_eigenspectrum(condensed: np.ndarray) -> np.ndarray:
    """Full eigenspectrum (descending) of the mean-imputed, double-centred
    Gram matrix -- classical (Torgerson) MDS, same construction as
    SpAM_Simulations/empirical/gt_construction.py's _classical_embed().

    CAVEAT (see that module's docstring): mean-imputing missing pairs before
    eigendecomposition is a known-invalid dimensionality estimate at this
    RDM's coverage level (~35-37% missing here, same order of magnitude as
    the ~63.6%-missing case that module measured) -- imputed pairs are
    mutually equidistant, which manufactures rank rather than reflecting
    real structure (a synthetic rank-8 ground truth reported an effective
    rank of ~193 under the identical mask+fill). This function is kept as a
    literal, qualitative eigenvalue scree only, matching what was asked for;
    it is NOT a validated dimensionality-selection method on this data.
    run_mds_sweep()'s SMACOF stress (no imputation) is the project's validated
    alternative for that purpose.
    """
    observed = ~np.isnan(condensed)
    imputed = np.where(observed, condensed, np.nanmean(condensed))
    gram = double_centered_gram(imputed)
    return np.linalg.eigvalsh(gram)[::-1]


def effective_rank(eigenvalues: np.ndarray, rel_tol: float = 1e-8) -> int:
    """Number of eigenvalues positive relative to the largest one -- the
    classical-MDS notion of "dimensionality" (rank of the Gram matrix). See
    classical_mds_eigenspectrum()'s caveat: on this data this count reflects
    the mean-imputation artifact more than the RDM's real structure."""
    threshold = rel_tol * eigenvalues[0]
    return int(np.sum(eigenvalues > threshold))


# ---------------------------------------------------------------------------
# Cached weighted-SMACOF sweep across dimensionalities
# ---------------------------------------------------------------------------


def run_mds_sweep(
    condensed: np.ndarray, dims: list[int], cache_key: str, cache_dir: Path,
    *, max_iters: int = 1000, convergence_tol: float = 1e-6,
) -> dict[int, dict]:
    """Weighted-SMACOF embeddings (R smacof::mds() via
    SpAM_Simulations.core.multi_dimensional_scaling.run_mds()) for each
    dimensionality in *dims*, one per-dimension .npz cache file under
    *cache_dir* (mds_{cache_key}_d{d}.npz) -- a real-scale fit here takes
    ~15-30s, so caching lets a re-run notebook skip already-fit dimensions
    rather than refitting all of them. NaN entries in *condensed* are
    treated as missing (weight 0) via condensed_rdm_to_weighted_dists().
    `max_iters` defaults to run_mds()'s own default (1000): at 500, some
    real fits on this project's data hit the iteration cap before
    `convergence_tol`.

    No default `cache_dir` -- that's a caller-specific concern (e.g. one
    directory per cohort/condition), not something this generic sweep should
    hardcode.

    Returns {ndim: {"conf": (n_conds, ndim) array, "confdist": condensed
    reconstructed-distance array (same length as *condensed*, but complete --
    no NaNs, since every pair has a fitted distance once embedded), "stress":
    float, "niter": float, "needs_more_iters": bool}} -- the last two let
    callers flag any dimension whose stress may not be fully converged.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dists, weights = condensed_rdm_to_weighted_dists(condensed)

    results: dict[int, dict] = {}
    run_mds = None
    for d in dims:
        cache_path = cache_dir / f"mds_{cache_key}_d{d}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            results[d] = {
                "conf": cached["conf"], "confdist": cached["confdist"], "stress": float(cached["stress"]),
                "niter": float(cached["niter"]), "needs_more_iters": bool(cached["needs_more_iters"]),
            }
            continue
        if run_mds is None:
            import os
            os.environ.setdefault("R_LIBS_USER", r"C:\Users\nirjo\R_library\4.5")
            from SpAM_Simulations.core.multi_dimensional_scaling import run_mds as _run_mds
            run_mds = _run_mds
        out = run_mds(dists=dists, weights=weights, ndim=d, max_iters=max_iters, convergence_tol=convergence_tol, precalc_init=True)
        conf = np.asarray(out["conf"], dtype=np.float64)
        confdist = np.asarray(out["confdist"], dtype=np.float64)
        stress = float(out["stress"])
        niter = float(out["niter"])
        needs_more_iters = bool(out["needs_more_iters"])
        np.savez(
            cache_path, conf=conf, confdist=confdist, stress=np.array(stress),
            niter=np.array(niter), needs_more_iters=np.array(needs_more_iters),
        )
        results[d] = {
            "conf": conf, "confdist": confdist, "stress": stress,
            "niter": niter, "needs_more_iters": needs_more_iters,
        }
    return results


# ---------------------------------------------------------------------------
# Comparing two fitted embeddings
# ---------------------------------------------------------------------------

_EQUIVALENT = {"disparity": 1e-3, "corr": 0.999}
_DIFFERENT = {"disparity": 0.05, "corr": 0.95}


def classify_mds_agreement(
    disparity: float, corr: float, *, equivalent: dict = _EQUIVALENT, different: dict = _DIFFERENT,
) -> str:
    """"EQUIVALENT" / "DIFFERENT" / "ambiguous" verdict from a Procrustes
    disparity + confdist correlation pair. Relocated from
    analysis/rsatoolbox_eval/mds_deep_dive.py's _classify()."""
    if disparity < equivalent["disparity"] and corr > equivalent["corr"]:
        return "EQUIVALENT"
    if disparity > different["disparity"] or corr < different["corr"]:
        return "DIFFERENT"
    return "ambiguous"


def compare_mds_configs(out_a: dict, out_b: dict, ndim: int) -> dict:
    """{"disparity", "confdist_r", "verdict"} between two run_mds() (or
    run_mds_sweep() per-dimension) outputs. Unifies
    analysis/rsatoolbox_eval/mds_deep_dive.py's _compare() scoring and
    analysis/behavioral/generalizability.py's _procrustes_compare()."""
    conf_a = np.asarray(out_a["conf"]).reshape(-1, ndim)
    conf_b = np.asarray(out_b["conf"]).reshape(-1, ndim)
    disparity = procrustes_m2(conf_a, conf_b)
    corr = float(np.corrcoef(np.asarray(out_a["confdist"]), np.asarray(out_b["confdist"]))[0, 1])
    return {"disparity": disparity, "confdist_r": corr, "verdict": classify_mds_agreement(disparity, corr)}
