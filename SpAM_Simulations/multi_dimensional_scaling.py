from typing import Dict, Any

import numpy as np
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import connected_components

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
        verbose: bool = False,
) -> Dict[str, Any]:
    assert ndim > 0, "`ndim` must be positive"
    dists = convert_to_condensed(dists)
    weights = convert_to_condensed(weights)
    assert dists.shape == weights.shape, "`dists` and `weights` must have the same shape"
    weights[(dists == 0) | (np.isnan(dists))] = 0.0     # insure missing distances have zero weight
    # convert to square form for smacof
    dists = squareform(dists, checks=False)
    weights = squareform(weights, checks=False)
    # check if the distance graph is fully connected
    n_components = connected_components(weights, directed=False, return_labels=False)
    if n_components > 1:
        raise RuntimeError(f"The distance graph has {n_components} connected components.")
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_dists = dists
        r_weights = weights
        res = smacof.mds(
            delta=r_dists, weightmat=r_weights, ndim=ndim, verbose=verbose, type="ratio", itmax=max_iters,
        )
        # convert from R's NamedList to Python's dict
        res_dict = {"max_iters": max_iters}
        for (key, val) in zip(res.names(), res.values()):
            val = val[0] if val.size == 1 else val
            key = str(key)
            if key.lower() == "call":
                # the R code that was generated to run the model
                continue
            res_dict[str(key)] = val.item() if val.size == 1 else val
        res_dict["needs_more_iters"] = res_dict["niter"] >= max_iters
        if res_dict["needs_more_iters"]:
            raise RuntimeError(f"MDS did not converge within {max_iters} iterations. Consider increasing `max_iters`.")
        return res_dict


