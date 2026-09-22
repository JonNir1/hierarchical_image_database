"""
Compare the current R-backed run_mds() (multi_dimensional_scaling.py) against an
rsatoolbox.util.weighted_mds-backed equivalent, on identical inputs.

This is a pre-swap validation tool for the plan section "Refactor MDS backend: R
smacof -> rsatoolbox.util.weighted_mds" -- it does not modify multi_dimensional_scaling.py.
run_mds_rsatoolbox() here mirrors run_mds()'s preprocessing (convert_to_condensed,
weight zeroing, connectivity check, optional sklearn-based precalculated init) exactly,
so any difference in the result comes from the SMACOF solver itself (R's smacof::mds
vs rsatoolbox's smacof()), not from preprocessing drift.

Two known, load-bearing differences between the two solvers (found by reading both
implementations' source, not assumed):

1. Non-convergence handling: R's smacof::mds() always returns, even if it hits
   itmax without meeting its internal eps criterion -- run_mds() detects this via
   niter >= max_iters and reports needs_more_iters=True with a still-usable conf.
   rsatoolbox.util.weighted_mds._smacof_single(), if it completes max_iter
   iterations without ever satisfying its own eps criterion, raises
   ValueError('No iterations, max_iter must be > 0') from a for/else -- with no
   partial configuration recoverable from that exception. run_mds_rsatoolbox()
   below catches this and reports needs_more_iters=True but conf=None/confdist=None
   (there is nothing to return). A real swap would need to either accept this
   (losing the partial embedding for non-converged runs, which the sweep's
   embedding-stability analysis currently uses) or reimplement the loop locally
   without the raise.

2. eps semantics: R's `eps` and rsatoolbox's `eps` are different convergence
   formulas (not the same threshold on the same quantity), so convergence_tol
   is passed through to both as-is but is not expected to produce matching
   iteration counts, even from an identical start.

precalc_init: SpAM_Simulations.core.config.MDSSweepConfig defaults to
precalc_init=False (confirmed by reading config.py), meaning the sweep's default
path starts both solvers from independent random inits -- not a controlled,
bit-comparable starting point. Only precalc_init=True gives both solvers the
identical starting configuration, which is the only case where "did they reach
the same fixed point" is a meaningful, checkable question; precalc_init=False is
still run here and compared, but only on stress magnitude and Procrustes fit
quality, not raw agreement.

Usage (from repo root, with the main checkout's .venv active and R_LIBS_USER set
to the user's R library so the R backend is importable):
    python -m analysis.rsatoolbox_eval.mds_rsatoolbox_compare
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.spatial import procrustes

os.environ.setdefault("R_LIBS_USER", r"C:\Users\nirjo\R_library\4.5")

from SpAM_Simulations.core.helpers import convert_to_condensed
from SpAM_Simulations.core.multi_dimensional_scaling import (
    run_mds, _precalculate_initial_embeddings,
)


def run_mds_rsatoolbox(
    dists: np.ndarray,
    weights: np.ndarray,
    ndim: int,
    max_iters: int = 1000,
    convergence_tol: float = 1e-6,
    precalc_init: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """rsatoolbox.util.weighted_mds-backed equivalent of run_mds(), same preprocessing.

    See the module docstring for the two known solver-level differences from the
    R backend (non-convergence handling, eps semantics).
    """
    from rsatoolbox.util.weighted_mds import smacof as rsa_smacof

    assert ndim > 0, "`ndim` must be positive"
    dists = convert_to_condensed(dists)
    weights = convert_to_condensed(weights)
    assert dists.shape == weights.shape, "`dists` and `weights` must have the same shape"
    weights[(dists == 0) | (np.isnan(dists))] = 0.0
    dists_sq = squareform(dists, checks=False)
    weights_sq = squareform(weights, checks=False)

    n_components = connected_components(weights_sq, directed=False, return_labels=False)
    if n_components > 1:
        raise RuntimeError(f"The distance graph has {n_components} connected components.")

    if precalc_init:
        precalc_conf = _precalculate_initial_embeddings(
            dists_sq=dists_sq, ndim=ndim,
            max_iters=min(50, max_iters // 10),
            convergence_tol=min(convergence_tol, 1e-3),
            num_runs=1, random_state=42, verbose=verbose,
        )
    else:
        precalc_conf = None

    try:
        conf, stress, niter = rsa_smacof(
            dists_sq, metric=True, n_components=ndim, init=precalc_conf,
            n_init=1, max_iter=max_iters, eps=convergence_tol,
            random_state=42, return_n_iter=True, weight=weights_sq,
        )
    except ValueError as e:
        if "No iterations" not in str(e):
            raise
        return {
            "max_iters": max_iters, "needs_more_iters": True, "niter": max_iters,
            "stress": np.nan, "conf": None, "confdist": None,
            "nonconvergence_raised": True,
        }

    return {
        "max_iters": max_iters, "needs_more_iters": bool(niter >= max_iters),
        "niter": niter, "stress": stress, "conf": conf, "confdist": pdist(conf),
        "nonconvergence_raised": False,
    }


def compare_one(
    label: str, dists: np.ndarray, weights: np.ndarray, ndim: int,
    max_iters: int, convergence_tol: float, precalc_init: bool,
) -> None:
    print(f"\n--- {label} (ndim={ndim}, max_iters={max_iters}, precalc_init={precalc_init}) ---")
    try:
        r_out = run_mds(dists, weights, ndim, max_iters, convergence_tol, precalc_init)
    except RuntimeError as e:
        print(f"  SKIP: {e}")
        return
    rsa_out = run_mds_rsatoolbox(dists, weights, ndim, max_iters, convergence_tol, precalc_init)

    print(f"  R:         niter={r_out['niter']:.0f}, stress={r_out['stress']:.6g}, "
          f"needs_more_iters={r_out['needs_more_iters']}")
    if rsa_out["nonconvergence_raised"]:
        print("  rsatoolbox: did not converge within max_iters -- smacof() RAISED "
              "(no partial config recoverable, see module docstring)")
        return
    print(f"  rsatoolbox: niter={rsa_out['niter']}, stress={rsa_out['stress']:.6g}, "
          f"needs_more_iters={rsa_out['needs_more_iters']}")

    r_conf = np.asarray(r_out["conf"]).reshape(-1, ndim)
    rsa_conf = np.asarray(rsa_out["conf"]).reshape(-1, ndim)
    _, _, disparity = procrustes(r_conf, rsa_conf)
    print(f"  Procrustes disparity (0 = identical up to rotation/reflection/scale): {disparity:.6g}")

    r_confdist = np.asarray(r_out["confdist"])
    rsa_confdist = np.asarray(rsa_out["confdist"])
    rho = float(np.corrcoef(r_confdist, rsa_confdist)[0, 1])
    print(f"  Pearson r between fitted distance vectors (confdist): {rho:.6f}")


def _synthetic_case():
    """Small synthetic connected weighted distance matrix, matching test_pipeline_mds.py's
    scale (~30 items) so this doesn't need a real simulation run."""
    rng = np.random.default_rng(7)
    n = 30
    true_ndim = 3
    coords = rng.normal(size=(n, true_ndim))
    dists = squareform(pdist(coords))  # square (n, n)
    noise = rng.normal(scale=0.05, size=dists.shape)
    noise = (noise + noise.T) / 2
    dists = np.abs(dists + noise)
    np.fill_diagonal(dists, 0.0)
    weights = np.ones_like(dists)
    return dists, weights


def _real_behavioral_case(variant_name: str):
    """Real behav_pre/behav_post RDM + a coverage-derived weight matrix (0 for
    never-observed pairs). Restricted to a dense, fully-connected sub-block since
    the full RDM (35% uncovered pairs) is not guaranteed connected as-is."""
    from analysis.rdms.common import load_rdm, RESULTS_DIR
    import json

    condensed = load_rdm(variant_name)
    sq = squareform(condensed)
    covered = ~np.isnan(sq)
    np.fill_diagonal(covered, True)
    degree = covered.sum(axis=1)
    keep = np.argsort(-degree)[:60]  # 60 best-covered images -> a small, likely-connected block
    sub = sq[np.ix_(keep, keep)]
    weights = (~np.isnan(sub)).astype(float)
    np.fill_diagonal(weights, 0.0)
    sub = np.nan_to_num(sub, nan=0.0)
    return sub, weights


def main() -> None:
    print("=" * 70)
    print("Synthetic case (n=30, true_ndim=3)")
    print("=" * 70)
    dists, weights = _synthetic_case()
    for precalc in (True, False):
        compare_one("synthetic", dists, weights, ndim=3, max_iters=300,
                    convergence_tol=1e-6, precalc_init=precalc)

    for variant_name in ("behav_pre", "behav_post"):
        try:
            dists, weights = _real_behavioral_case(variant_name)
        except FileNotFoundError:
            print(f"\n{variant_name} not built yet -- skipping.")
            continue
        print("\n" + "=" * 70)
        print(f"Real data: {variant_name} (60 best-covered images subset)")
        print("=" * 70)
        for precalc in (True, False):
            compare_one(variant_name, dists, weights, ndim=5, max_iters=300,
                        convergence_tol=1e-6, precalc_init=precalc)


if __name__ == "__main__":
    main()
