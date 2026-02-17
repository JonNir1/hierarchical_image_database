from typing import Dict, Any, Optional

import numpy as np
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import MDS

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

from SpAM_Simulations.helpers import convert_to_condensed


# import the R package `smacof` for multidimensional scaling
try:
    smacof = importr('smacof')
except ImportError:
    raise ImportError("Please install 'smacof' in R first: install.packages('smacof')")


def run_mds(
        dists: np.ndarray,
        weights: np.ndarray,
        ndim: int,
        max_iters: int = 1000,
        convergence_tol: float = 1e-6,
        precalc_init: bool = True,
        verbose: bool = False,
) -> Dict[str, Any]:
    assert ndim > 0, "`ndim` must be positive"
    dists = convert_to_condensed(dists)
    weights = convert_to_condensed(weights)
    assert dists.shape == weights.shape, "`dists` and `weights` must have the same shape"
    # insure missing distances have zero weight
    weights[(dists == 0) | (np.isnan(dists))] = 0.0
    # convert to square form for smacof
    dists = squareform(dists, checks=False)
    weights = squareform(weights, checks=False)
    # check if the distance graph is fully connected
    n_components = connected_components(weights, directed=False, return_labels=False)
    if n_components > 1:
        raise RuntimeError(f"The distance graph has {n_components} connected components.")
    if precalc_init:
        # run a fast non-metric MDS to generate an initial estimate for the metric MDS
        precalc_conf = _precalculate_initial_embeddings(
            dists_sq=dists,
            ndim=ndim,
            max_iters=min(50, max_iters // 10),             # low number of iterations, just to get initial estimate
            convergence_tol=min(convergence_tol, 1e-3),     # loose convergence threshold, just to get initial estimate
            num_runs=1, random_state=42, verbose=verbose
        )
    else:
        precalc_conf = None
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_mds_kwargs = {
            "ndim": ndim, "itmax": max_iters, "eps": convergence_tol, "type": "ratio", "verbose": verbose,
            "init": "random" if precalc_conf is None else precalc_conf
        }
        r_dists = dists
        r_weights = weights
        res = smacof.mds(delta=r_dists, weightmat=r_weights, **r_mds_kwargs)
        # convert from R's NamedList to Python's dict
        res_dict = {"max_iters": max_iters}
        for (key, val) in zip(res.names(), res.values()):
            if key.lower() == "call":
                # the R code that was generated to run the model
                continue
            res_dict[str(key)] = val.item() if val.size == 1 else val
        res_dict["needs_more_iters"] = res_dict["niter"] >= max_iters
        return res_dict


def _precalculate_initial_embeddings(
        dists_sq: np.ndarray,
        ndim: int,
        max_iters: int,
        convergence_tol: float,
        num_runs: int = 1,
        random_state: int = 42,
        verbose: bool = False
) -> Optional[np.ndarray]:
    if not np.allclose(dists_sq, dists_sq.T, atol=convergence_tol, rtol=0, equal_nan=True):
        raise ValueError("dists_sq is not a symmetric matrix")
    try:
        precalc_model = MDS(
            n_components=ndim,
            max_iter=max_iters,
            eps=convergence_tol,
            n_init=num_runs,
            random_state=random_state,
            n_jobs=1,
            metric_mds=False,
            init="random",
            metric="precomputed",
        )
        precalc_conf = precalc_model.fit_transform(dists_sq)
        return precalc_conf
    except Exception as e:
        if verbose:
            print(f"Failed to precalculate initial embeddings: {e}")
        return None
