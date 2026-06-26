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
from SpAM_Simulations.task_v2_3_experiment import TaskV2_3ExperimentParameters
from SpAM_Simulations.task_v2_4_experiment import TaskV2_4ExperimentParameters
from SpAM_Simulations.task_v3_experiment import TaskV3ExperimentParameters


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
class TaskV2_3SimulationConfig(SimulationConfig):
    """``SimulationConfig`` extended with the real task's image-repetition design lever.

    Adds ``frac_images_repeated`` - the fraction of each subject's active image subset
    shown in 2 trials instead of 1, swept like the other grids (see ``design.py``). The
    GT-source fields, ``uses_random_ground_truth``, and the GT-source validation are
    inherited unchanged from ``SimulationConfig``.
    """
    frac_images_repeated: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self):
        super().__post_init__()
        if len(self.frac_images_repeated) == 0:
            raise ValueError("parameter grid(s) must be non-empty: ['frac_images_repeated']")

    def param_grid(self) -> List[TaskV2_3ExperimentParameters]:
        return [
            TaskV2_3ExperimentParameters(*p)
            for p in product(
                self.num_subjects,
                self.trials_per_subject,
                self.images_per_trial,
                self.subjects_noise_scale,
                self.subjects_noise_df,
                self.frac_images_repeated,
            )
        ]


@dataclass
class TaskV2_4SimulationConfig(TaskV2_3SimulationConfig):
    """``TaskV2_3SimulationConfig`` extended with the real task's whole-trial-repeat lever.

    Adds ``frac_trials_repeated`` - the fraction of each subject's ``trials_per_subject`` slots
    shown again verbatim (test-retest reliability), swept like the other grids. Inherits
    ``frac_images_repeated``, the GT-source fields, and all validation from
    ``TaskV2_3SimulationConfig``.
    """
    frac_trials_repeated: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self):
        super().__post_init__()
        if len(self.frac_trials_repeated) == 0:
            raise ValueError("parameter grid(s) must be non-empty: ['frac_trials_repeated']")
        bad = [fr for fr in self.frac_trials_repeated if not (0 <= fr < 1)]
        if bad:
            raise ValueError(f"`frac_trials_repeated` values must be in [0, 1), got {bad}")

    def param_grid(self) -> List[TaskV2_4ExperimentParameters]:
        return [
            TaskV2_4ExperimentParameters(*p)
            for p in product(
                self.num_subjects,
                self.trials_per_subject,
                self.images_per_trial,
                self.subjects_noise_scale,
                self.subjects_noise_df,
                self.frac_images_repeated,
                self.frac_trials_repeated,
            )
        ]


@dataclass
class TaskV3SimulationConfig(SimulationConfig):
    """``SimulationConfig`` for the task-v3 generative (coordinate-space) model.

    Inherits from the **base** config, not the v2.3/v2.4 ones, because task v3.0 drops the
    ``frac_images_repeated`` cross-context lever entirely (see ``task_v3_experiment``). Adds:

    * ``frac_trials_repeated`` - swept whole-trial-repeat fraction (test-retest reliability).
    * ``perspective_dispersion`` - swept dispersion of each subject's per-PC weight vector
      (between-subject "perspective" disagreement; 0 = everyone shares the ground-truth geometry).
    * ``use_isotropic`` / ``decay`` / ``n_clusters`` - ground-truth *spectrum* controls, consumed by
      ``simulation.build_ground_truth_embeddings`` at generation time (not swept). ``use_isotropic``
      headlines the conservative full-rank case; ``False`` (geometric ``decay``, optional
      hierarchical ``n_clusters``) is the realistic case.

    The ground truth must be synthetic here (``n_images`` + ``n_dims``); the spectrum controls have
    no meaning for a supplied ``gt_embeddings``.
    """
    frac_trials_repeated: Sequence[float] = field(default_factory=tuple)
    perspective_dispersion: Sequence[float] = field(default_factory=tuple)
    use_isotropic: bool = True
    decay: float = 0.7
    n_clusters: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        empty = [name for name in ("frac_trials_repeated", "perspective_dispersion")
                 if len(getattr(self, name)) == 0]
        if empty:
            raise ValueError(f"parameter grid(s) must be non-empty: {empty}")
        bad = [fr for fr in self.frac_trials_repeated if not (0 <= fr < 1)]
        if bad:
            raise ValueError(f"`frac_trials_repeated` values must be in [0, 1), got {bad}")
        if any(pd < 0 for pd in self.perspective_dispersion):
            raise ValueError("`perspective_dispersion` values must be non-negative")
        if not self.uses_random_ground_truth:
            raise ValueError(
                "TaskV3SimulationConfig requires synthetic ground truth (n_images + n_dims); the "
                "spectrum controls (use_isotropic/decay/n_clusters) don't apply to gt_embeddings"
            )

    def param_grid(self) -> List[TaskV3ExperimentParameters]:
        return [
            TaskV3ExperimentParameters(*p)
            for p in product(
                self.num_subjects,
                self.trials_per_subject,
                self.images_per_trial,
                self.subjects_noise_scale,
                self.subjects_noise_df,
                self.frac_trials_repeated,
                self.perspective_dispersion,
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
