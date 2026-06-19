"""Declarative configuration for SpAM simulations and MDS sweeps.

These dataclasses make it easy to spin up a new simulation with a different configuration
without editing notebook cells: build a ``SimulationConfig`` (random or real-data ground
truth + a parameter grid) and a ``MDSSweepConfig`` (which target dimensionalities to fit),
then hand them to the functions in ``pipeline.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Sequence

import numpy as np

from SpAM_Simulations.experiment import ExperimentParameters


@dataclass
class SimulationConfig:
    """Specifies a simulation: a ground-truth embedding plus a grid of experiment parameters.

    Ground truth is either random (``n_images`` + ``n_dims``) or supplied directly via
    ``gt_embeddings`` (e.g. real image features) - exactly one of the two must be given.
    Each list-valued field is swept; the Cartesian product defines the configurations, each
    run ``reps`` times.
    """
    num_subjects: Sequence[int]
    trials_per_subject: Sequence[int]
    images_per_trial: Sequence[int]
    subjects_noise_scale: Sequence[float]
    subjects_noise_df: Sequence[int]
    n_images: Optional[int] = None
    n_dims: Optional[int] = None
    gt_embeddings: Optional[np.ndarray] = None
    reps: int = 1
    seed: int = 42

    def __post_init__(self):
        has_random = self.n_images is not None and self.n_dims is not None
        has_embeddings = self.gt_embeddings is not None
        if has_random == has_embeddings:
            raise ValueError(
                "Specify exactly one ground-truth source: either (n_images, n_dims) or gt_embeddings"
            )
        grids = {
            "num_subjects": self.num_subjects,
            "trials_per_subject": self.trials_per_subject,
            "images_per_trial": self.images_per_trial,
            "subjects_noise_scale": self.subjects_noise_scale,
            "subjects_noise_df": self.subjects_noise_df,
        }
        empty = [name for name, vals in grids.items() if len(vals) == 0]
        if empty:
            raise ValueError(f"parameter grid(s) must be non-empty: {empty}")
        if self.reps <= 0:
            raise ValueError(f"`reps` must be positive (got {self.reps})")

    @property
    def uses_random_ground_truth(self) -> bool:
        return self.gt_embeddings is None

    def param_grid(self) -> List[ExperimentParameters]:
        """All experiment-parameter combinations from the Cartesian product of the grids."""
        return [
            ExperimentParameters(*p)
            for p in product(
                self.num_subjects,
                self.trials_per_subject,
                self.images_per_trial,
                self.subjects_noise_scale,
                self.subjects_noise_df,
            )
        ]


@dataclass
class MDSSweepConfig:
    """Specifies which target dimensionalities to fit and the SMACOF solver settings.

    By default fits every dimension from ``min_ndim`` up to the ground-truth dimensionality;
    pass an explicit ``ndims`` list to override.
    """
    min_ndim: int = 2
    ndims: Optional[Sequence[int]] = None
    max_iters: int = 500
    convergence_tol: float = 1e-6
    precalc_init: bool = False

    def target_dims(self, gt_dimensions: int) -> List[int]:
        if self.ndims is not None:
            dims = list(self.ndims)
        else:
            dims = list(range(self.min_ndim, gt_dimensions + 1))
        if not dims:
            raise ValueError(
                f"no target dimensions (min_ndim={self.min_ndim}, gt_dimensions={gt_dimensions})"
            )
        if any(d <= 0 for d in dims):
            raise ValueError(f"target dimensions must be positive, got {dims}")
        return dims
