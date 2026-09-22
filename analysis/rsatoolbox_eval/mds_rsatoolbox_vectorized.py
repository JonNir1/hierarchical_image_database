"""
A faithful, vectorized reimplementation of rsatoolbox.util.weighted_mds's weighted
SMACOF update -- same algorithm (including its known weight-blind stress/convergence
bug, see mds_rsatoolbox_compare.py's module docstring), just fast.

Why this exists: rsatoolbox's `_smacof_single()` builds the (n, n) "V" matrix used in
its weighted Guttman-transform update via a Python double loop over every unordered
pair, each iteration allocating and outer-producting a full (n, n) array:

    V = np.zeros((n_samples, n_samples))
    for nn in range(n_samples):
        for mm in range(nn, n_samples):
            v = np.zeros((n_samples, 1)); v[nn], v[mm] = 1, -1
            V += weight[nn, mm] * np.dot(v, v.T)

That's O(n**2) Python-level iterations each doing O(n**2) work: O(n**4) overall. It
didn't finish for n=500 within 2+ minutes in testing (killed); this project's real
behavioral RDMs are n=725. Separately from the stress-formula bug already found, this
makes rsatoolbox's weighted MDS unusable at this project's actual scale.

The fix: V is exactly the weight matrix's graph Laplacian. Expanding the double loop by
hand (see derivation below) shows V[i,i] = sum_j weight[i,j] (row sum) and V[i,j] = -weight[i,j]
for i != j -- i.e. V = diag(weight.sum(axis=1)) - weight, computable in O(n**2) with plain
numpy, no loop. `_laplacian()` below IS that identity; `_smacof_single_vectorized()` is
`_smacof_single()` with only the V construction replaced -- every other line (disparities,
stress, B matrix, pinv, the eps-based break condition, including its bug) is unchanged, so
this is a performance fix only, not a behavior change. Verified bit-close (not just
"similar") against the real rsatoolbox implementation at n small enough for both to run --
see __tests__ or the __main__ block below.

Derivation of V = diag(weight.sum(axis=1)) - weight:
For nn < mm, v = e_nn - e_mm (as a column vector), so v @ v.T has +1 at (nn,nn) and (mm,mm),
-1 at (nn,mm) and (mm,nn), 0 elsewhere. Summing weight[nn,mm] * v @ v.T over every unordered
pair {nn,mm} (nn < mm; the nn==mm loop entries contribute nothing given the zero diagonal
convention `run_mds`/`convert_to_condensed` already enforce) accumulates, per row i,
weight[i,j] into V[i,i] for every j != i (giving the row sum on the diagonal) and -weight[i,j]
into the off-diagonal V[i,j] -- exactly the (negative, off-diagonal-flipped) graph Laplacian.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, check_symmetric


def _laplacian(weight: np.ndarray) -> np.ndarray:
    """O(n**2) equivalent of rsatoolbox._smacof_single's O(n**4) double-loop V construction."""
    return np.diag(weight.sum(axis=1)) - weight


def smacof_single_vectorized(
    dissimilarities: np.ndarray, metric: bool = True, n_components: int = 2,
    init: Optional[np.ndarray] = None, max_iter: int = 300, verbose: int = 0,
    eps: float = 1e-3, random_state=None, weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """Line-for-line match of rsatoolbox.util.weighted_mds._smacof_single(), with only
    the V-matrix construction vectorized (see module docstring). Same signature, same
    return value, same bugs (weight-blind stress; raises on non-convergence)."""
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)
    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        X = random_state.rand(n_samples * n_components).reshape((n_samples, n_components))
    else:
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" % (n_samples, n_components))
        X = init

    old_stress = None
    ir = IsotonicRegression()
    for it in range(max_iter):
        dis = euclidean_distances(X)

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            dis_flat_w = dis_flat[sim_flat != 0]
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) / (disparities ** 2).sum())

        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        dis2 = dis.copy()
        dis2[dis2 == 0] = 1e-5
        if weight is None:
            ratio = disparities / dis2
            B = -ratio
            B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
            X = 1.0 / n_samples * np.dot(B, X)
        else:
            ratio = weight * disparities / dis2
            B = -ratio
            B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
            V = _laplacian(weight)  # was: O(n**4) double loop; see module docstring
            X = np.dot(np.linalg.pinv(V), np.dot(B, X))

        dis_sum = np.sqrt((X ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print(f"it: {it}, stress {stress}")
        if old_stress is not None:
            if (old_stress - stress / dis_sum) < eps:
                if verbose:
                    print(f"breaking at iteration {it} with stress {stress}")
                break
        old_stress = stress / dis_sum
    else:
        raise ValueError("No iterations, max_iter must be > 0")

    return X, stress, it + 1


def smacof_vectorized(
    dissimilarities: np.ndarray, *, metric: bool = True, n_components: int = 2,
    init: Optional[np.ndarray] = None, n_init: int = 8, max_iter: int = 300,
    verbose: int = 0, eps: float = 1e-3, random_state=None,
    return_n_iter: bool = False, weight: Optional[np.ndarray] = None,
):
    """Vectorized match of rsatoolbox.util.weighted_mds.smacof() (the multi-init wrapper).
    Always runs single-job (this project's use case is one run at a time, in a joblib
    worker -- see multi_dimensional_scaling.py); drops rsatoolbox's n_jobs/Parallel path."""
    from sklearn.utils import check_array

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    if hasattr(init, "__array__"):
        init = np.asarray(init).copy()
        n_init = 1

    best_pos, best_stress, best_iter = None, None, None
    for _ in range(n_init):
        pos, stress, n_iter_ = smacof_single_vectorized(
            dissimilarities, metric=metric, n_components=n_components, init=init,
            max_iter=max_iter, verbose=verbose, eps=eps, random_state=random_state,
            weight=weight,
        )
        if best_stress is None or stress < best_stress:
            best_stress, best_pos, best_iter = stress, pos.copy(), n_iter_

    if return_n_iter:
        return best_pos, best_stress, best_iter
    return best_pos, best_stress


def run_mds_rsatoolbox_vectorized(
    dists: np.ndarray, weights: np.ndarray, ndim: int, max_iters: int = 1000,
    convergence_tol: float = 1e-6, precalc_init: bool = True, verbose: bool = False,
) -> dict:
    """Same interface/preprocessing as mds_rsatoolbox_compare.run_mds_rsatoolbox(), but
    solving via smacof_vectorized() above instead of rsatoolbox's real (O(n**4), unusable
    at this project's n=725 scale) smacof(). Verified bit-close to the original at n=60
    -- see this module's __main__ block."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import connected_components

    from SpAM_Simulations.core.helpers import convert_to_condensed
    from SpAM_Simulations.core.multi_dimensional_scaling import _precalculate_initial_embeddings

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
            dists_sq=dists_sq, ndim=ndim, max_iters=min(50, max_iters // 10),
            convergence_tol=min(convergence_tol, 1e-3), num_runs=1, random_state=42,
            verbose=verbose,
        )
    else:
        precalc_conf = None

    try:
        conf, stress, niter = smacof_vectorized(
            dists_sq, metric=True, n_components=ndim, init=precalc_conf, n_init=1,
            max_iter=max_iters, eps=convergence_tol, random_state=42,
            return_n_iter=True, weight=weights_sq,
        )
    except ValueError as e:
        if "No iterations" not in str(e):
            raise
        return {
            "max_iters": max_iters, "needs_more_iters": True, "niter": max_iters,
            "stress": np.nan, "conf": None, "confdist": None, "nonconvergence_raised": True,
        }

    return {
        "max_iters": max_iters, "needs_more_iters": bool(niter >= max_iters),
        "niter": niter, "stress": stress, "conf": conf, "confdist": pdist(conf),
        "nonconvergence_raised": False,
    }


if __name__ == "__main__":
    # Equivalence smoke test against the real rsatoolbox implementation, at an n small
    # enough for the original O(n**4) path to still finish in reasonable time.
    import time
    from rsatoolbox.util.weighted_mds import smacof as rsa_smacof

    rng = np.random.default_rng(0)
    n = 60
    coords = rng.normal(size=(n, 5))
    dist = euclidean_distances(coords)
    weight = (rng.random((n, n)) < 0.65).astype(float)
    weight = np.triu(weight, 1)
    weight = weight + weight.T
    dist_masked = dist.copy()
    dist_masked[weight == 0] = 0.0
    init = coords + rng.normal(scale=0.01, size=coords.shape)

    t0 = time.time()
    conf_orig, stress_orig, niter_orig = rsa_smacof(
        dist_masked, n_components=5, weight=weight, init=init.copy(),
        n_init=1, max_iter=300, eps=1e-9, random_state=0, return_n_iter=True,
    )
    t_orig = time.time() - t0

    t0 = time.time()
    conf_vec, stress_vec, niter_vec = smacof_vectorized(
        dist_masked, n_components=5, weight=weight, init=init.copy(),
        n_init=1, max_iter=300, eps=1e-9, random_state=0, return_n_iter=True,
    )
    t_vec = time.time() - t0

    print(f"original:   niter={niter_orig}, stress={stress_orig:.10f}, time={t_orig:.3f}s")
    print(f"vectorized: niter={niter_vec}, stress={stress_vec:.10f}, time={t_vec:.3f}s")
    print(f"max |conf diff| = {np.abs(conf_orig - conf_vec).max():.3e}")
    print(f"niter match: {niter_orig == niter_vec}")
    assert niter_orig == niter_vec, "iteration counts diverge -- not a faithful port"
    np.testing.assert_allclose(conf_orig, conf_vec, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(stress_orig, stress_vec, rtol=1e-8)
    print("EQUIVALENCE CONFIRMED (rtol=1e-8)")
