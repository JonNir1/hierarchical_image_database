"""Shared builders for the cluster-analysis tests: planted blobs and the ResultStores made from them.

Not a conftest on purpose: tests import it explicitly (``from _store_helpers import ...``), which
works because the test directory is on ``sys.path`` (there is no ``__init__.py``).
"""
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist

from SpAM_Simulations.core.storage import ResultStore

MAX_NDIM = 4
CLUSTER_N_IMAGES = 30
DENSITY_N_IMAGES = 60
ARMS_N_IMAGES = 45
META = ["num_subjects", "rep", "ndim", "niter", "stress", "status"]
ARMS_META = ["num_subjects", "allocation_mode", "rep", "ndim", "niter", "stress", "status"]


def blobs(n_per: int = 20, k: int = 3, sd: float = 0.25, seed: int = 0, ndim: int = 3,
          centre_sd: float = 10.0, dtype: Optional[type] = None) -> np.ndarray:
    """`k` well-separated Gaussian blobs; the planted structure the metrics should recover."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, centre_sd, (k, ndim))
    out = np.vstack([c + rng.normal(0, sd, (n_per, ndim)) for c in centres])
    return out if dtype is None else out.astype(dtype)


def blobs_with_outliers(n_per: int = 18, k: int = 3, n_outliers: int = 6, sd: float = 0.2,
                        seed: int = 0, ndim: int = MAX_NDIM) -> np.ndarray:
    """``k`` tight blobs plus ``n_outliers`` scattered points that belong to none of them."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 10, size=(k, ndim))
    pts = [c + rng.normal(0, sd, size=(n_per, ndim)) for c in centres]
    pts.append(rng.normal(0, 25, size=(n_outliers, ndim)))
    return np.vstack(pts).astype(np.float32)


def append_fits(store: ResultStore, truth: np.ndarray, groups: Sequence[Mapping],
                n_reps: int, noise_sd: float, rng: np.random.Generator, ndim: int,
                statuses: Optional[Mapping[tuple, str]] = None) -> None:
    """Append `n_reps` noisy views of `truth` per group; `statuses` is keyed (group index, rep)."""
    for gi, group in enumerate(groups):
        for rep in range(n_reps):
            status = (statuses or {}).get((gi, rep), "success")
            meta = {**group, "rep": rep, "ndim": ndim, "niter": 10, "stress": 0.1,
                    "status": status}
            if status not in ("success", "max_iters"):
                store.append(meta)
                continue
            coords = (truth + rng.normal(0, noise_sd, truth.shape)).astype(np.float32)
            store.append(meta, confdist=pdist(coords).astype(np.float32), conf=coords)


def build_store(path: Path, truth: np.ndarray, groups: Sequence[Mapping], n_reps: int,
                noise_sd: float, meta_columns: Sequence[str], seed: int, ndim: int,
                statuses: Optional[Mapping[tuple, str]] = None) -> ResultStore:
    """Create a store at `path` whose cohorts are noisy views of one planted configuration."""
    n_images, max_ndim = truth.shape
    store = ResultStore.create(path, confdist_len=n_images * (n_images - 1) // 2,
                               meta_columns=list(meta_columns), n_images=n_images,
                               max_ndim=max_ndim)
    append_fits(store, truth, groups, n_reps, noise_sd, np.random.default_rng(seed), ndim,
                statuses)
    store.close()
    return ResultStore.open(path)


def conf_store(root: Path, n_reps: int = 4, num_subjects: Sequence[int] = (20, 50),
               ndim: int = 3, statuses: Optional[Mapping[tuple, str]] = None,
               seed: int = 0) -> ResultStore:
    """A store whose cohorts are noisy views of one planted 3-blob structure."""
    truth = blobs(n_per=CLUSTER_N_IMAGES // 3, k=3, sd=0.25, seed=seed, ndim=MAX_NDIM)
    return build_store(root / "s", truth, [{"num_subjects": ns} for ns in num_subjects],
                       n_reps, 0.1, META, seed, ndim, statuses)


def append_duplicate_reps(store_path: Path, n_reps: int, num_subjects: Sequence[int],
                          ndim: int = 3, seed: int = 99) -> ResultStore:
    """Re-append a second copy of every fit, as a resume that lost track of completed work does."""
    truth = blobs(n_per=CLUSTER_N_IMAGES // 3, k=3, sd=0.25, seed=0, ndim=MAX_NDIM)
    store = ResultStore.open(store_path)
    append_fits(store, truth, [{"num_subjects": ns} for ns in num_subjects], n_reps, 0.1,
                np.random.default_rng(seed), ndim)
    store.close()
    return ResultStore.open(store_path)


def density_store(root: Path, n_reps: int = 3, seed: int = 0) -> ResultStore:
    """Noisy views of blobs plus planted outliers, for the density pass."""
    return build_store(root / "s", blobs_with_outliers(seed=seed), [{"num_subjects": 50}],
                       n_reps, 0.05, META, seed, MAX_NDIM)


def arms_store(root: Path, n_reps: int = 3, arms: Sequence[float] = (0.0, 1.0),
               sd: float = 0.25, seed: int = 0) -> Path:
    """A two-arm store of noisy realisations of the same three planted blobs; returns its path."""
    truth = blobs(ARMS_N_IMAGES // 3, 3, sd, seed, MAX_NDIM, centre_sd=6.0, dtype=np.float32)
    groups = [{"num_subjects": 50, "allocation_mode": arm} for arm in arms]
    return build_store(root / "s", truth, groups, n_reps, 0.1, ARMS_META, seed, MAX_NDIM).path
