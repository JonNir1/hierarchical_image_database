"""
Deep-dive 2x2 comparison: aggregation method (mean/evidence) x MDS backend
(R smacof::mds() / rsatoolbox weighted_mds), per the plan's section 4.2.

The 4 conditions, for each of behav_pre/behav_post:
    C1 = mean + R        C3 = evidence + R
    C2 = mean + rsatoolbox (vectorized -- see mds_rsatoolbox_vectorized.py;
                              rsatoolbox's own smacof() is O(n**4), unusable at n=725)
    C4 = evidence  + rsatoolbox (vectorized)

Weight matrix pinned to binary was_observed (from each RDM's own NaN pattern) for all
four conditions, so the comparison isolates the two factors above rather than conflating
either with a third, weighting-scheme axis.

Usage (from repo root, main checkout's .venv, R_LIBS_USER set):
    python -m analysis.rsatoolbox_eval.mds_deep_dive
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform

os.environ.setdefault("R_LIBS_USER", r"C:\Users\nirjo\R_library\4.5")

from analysis.rdms.common import load_rdm
from analysis.rsatoolbox_eval.mds_rsatoolbox_vectorized import run_mds_rsatoolbox_vectorized
from SpAM_Simulations.core.multi_dimensional_scaling import run_mds

_EQUIVALENT = {"disparity": 1e-3, "corr": 0.999}
_DIFFERENT = {"disparity": 0.05, "corr": 0.95}

_AGG_METHODS = ("mean", "evidence")
_DIMS = (2, 3, 5, 10)
_MAX_ITERS = 500
_CONVERGENCE_TOL = 1e-6


def _load_condition_inputs(variant: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Real RDM -> (dists with NaN->0 placeholder, binary was_observed weight)."""
    d = load_rdm(f"behav_{variant}_{method}")
    sq = squareform(d)
    weight = (~np.isnan(sq)).astype(np.float64)
    np.fill_diagonal(weight, 0.0)
    dists = np.nan_to_num(sq, nan=0.0)
    return dists, weight


def _run_condition(backend: str, dists, weights, ndim, eps) -> dict | None:
    try:
        if backend == "R":
            return run_mds(dists, weights, ndim, _MAX_ITERS, eps, precalc_init=True, verbose=False)
        out = run_mds_rsatoolbox_vectorized(dists, weights, ndim, _MAX_ITERS, eps, precalc_init=True, verbose=False)
        return None if out.get("nonconvergence_raised") else out
    except RuntimeError as e:
        print(f"    SKIP ({backend}): {e}")
        return None


def _classify(disparity: float, corr: float) -> str:
    if disparity < _EQUIVALENT["disparity"] and corr > _EQUIVALENT["corr"]:
        return "EQUIVALENT"
    if disparity > _DIFFERENT["disparity"] or corr < _DIFFERENT["corr"]:
        return "DIFFERENT"
    return "ambiguous"


def _compare(label_a, out_a, label_b, out_b, ndim) -> None:
    if out_a is None or out_b is None:
        print(f"    {label_a} vs {label_b}: SKIP (one or both conditions unavailable)")
        return
    conf_a = np.asarray(out_a["conf"]).reshape(-1, ndim)
    conf_b = np.asarray(out_b["conf"]).reshape(-1, ndim)
    _, _, disparity = procrustes(conf_a, conf_b)
    corr = float(np.corrcoef(np.asarray(out_a["confdist"]), np.asarray(out_b["confdist"]))[0, 1])
    verdict = _classify(disparity, corr)
    print(f"    {label_a} vs {label_b}: disparity={disparity:.6g}, confdist_r={corr:.6f}  -> {verdict}")


def run_cell(variant: str, ndim: int) -> dict:
    print(f"\n--- variant={variant}, ndim={ndim} ---")
    conditions = {}
    for method in _AGG_METHODS:
        dists, weights = _load_condition_inputs(variant, method)
        for backend, key in (("R", f"{method}_R"), ("rsatoolbox", f"{method}_rsa")):
            out = _run_condition(backend, dists, weights, ndim, _CONVERGENCE_TOL)
            conditions[key] = out
            if out is not None:
                print(f"    {key}: niter={out['niter']}, stress={out['stress']:.6g}, "
                      f"needs_more_iters={out['needs_more_iters']}")

    c1, c2, c3, c4 = conditions["mean_R"], conditions["mean_rsa"], conditions["evidence_R"], conditions["evidence_rsa"]
    print("  Backend effect (aggregation fixed):")
    _compare("C1(mean+R)", c1, "C2(mean+rsatoolbox)", c2, ndim)
    _compare("C3(evidence+R)", c3, "C4(evidence+rsatoolbox)", c4, ndim)
    print("  Aggregation effect (backend fixed):")
    _compare("C1(mean+R)", c1, "C3(evidence+R)", c3, ndim)
    _compare("C2(mean+rsatoolbox)", c2, "C4(evidence+rsatoolbox)", c4, ndim)
    print("  Both factors (completeness):")
    _compare("C1(mean+R)", c1, "C4(evidence+rsatoolbox)", c4, ndim)
    _compare("C2(mean+rsatoolbox)", c2, "C3(evidence+R)", c3, ndim)

    return conditions


def eps_sensitivity_check(variant: str, method: str, ndim: int) -> None:
    """Bug-signature check (part of the minimal-repro search): run the rsatoolbox-backed
    condition at two very different eps values and flag if niter is identical -- the
    signature found in the original real-data trace (see plan section 4.1)."""
    dists, weights = _load_condition_inputs(variant, method)
    results = {}
    for eps in (1e-6, 1e-12):
        out = _run_condition("rsatoolbox", dists, weights, ndim, eps)
        results[eps] = out["niter"] if out is not None else None
    same = results[1e-6] is not None and results[1e-6] == results[1e-12]
    flag = " <-- BUG SIGNATURE (eps has no effect)" if same else ""
    print(f"    eps-sensitivity [{variant}/{method}/ndim={ndim}]: "
          f"niter(eps=1e-6)={results[1e-6]}, niter(eps=1e-12)={results[1e-12]}{flag}")


def main() -> None:
    for variant in ("pre", "post"):
        for ndim in _DIMS:
            run_cell(variant, ndim)

    print("\n" + "=" * 70)
    print("Systematic eps-sensitivity sweep (minimal-repro search)")
    print("=" * 70)
    for variant in ("pre", "post"):
        for method in _AGG_METHODS:
            for ndim in _DIMS:
                eps_sensitivity_check(variant, method, ndim)


if __name__ == "__main__":
    main()
