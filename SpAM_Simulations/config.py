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
from SpAM_Simulations.task_v4_experiment import TaskV4ExperimentParameters


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
        # GT may be synthetic (n_images + n_dims, using use_isotropic/decay/n_clusters) OR a supplied
        # `gt_embeddings` (e.g. the pilot-calibrated embedding); the base class already enforces that
        # exactly one source is given. With gt_embeddings the spectrum controls are simply ignored.

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
class TaskV4SimulationConfig(TaskV3SimulationConfig):
    """``TaskV3SimulationConfig`` extended with the deployed v4.0 **screening block**.

    Adds three swept levers (see ``task_v4_experiment``):

    * ``screening_trials`` - screening-stage main-trial slots per candidate (deployed: 8);
      ``0`` skips the block entirely and reduces the model to task-v3.
    * ``screening_repeats`` - how many of those are verbatim repeats (deployed: 2). Must leave at
      least one original, so it is swept jointly with ``screening_trials`` in practice.
    * ``screening_min_reliability`` - exclude a candidate whose *minimum* per-repeat test-retest
      Spearman falls below this (deployed: 0.0). ``-1.0`` runs the block but excludes nobody,
      which is the right control arm: it isolates the effect of *exclusion* while holding the
      number of collected trials constant.

    Ground truth may be synthetic or a supplied ``gt_embeddings`` (the pilot-calibrated
    embedding), exactly as for task-v3.
    """
    screening_trials: Sequence[int] = field(default_factory=tuple)
    screening_repeats: Sequence[int] = field(default_factory=tuple)
    screening_min_reliability: Sequence[float] = field(default_factory=tuple)
    # Shape of the per-subject noise population. >0 selects a lognormal with that sigma; 0.0 falls
    # back to |t(subjects_noise_df)|, the historical family. Defaults to (0.0,) so an existing
    # config keeps its exact meaning. See `noise_population` for why the t family is inadequate:
    # its CV cannot go below ~0.756, but the pilot's reliability distribution needs ~0.47.
    subjects_noise_lognormal_sigma: Sequence[float] = (0.0,)
    # Image-to-trial allocation arm: 0.0 = random (what the deployed task does and what every
    # previous run used), 1.0 = balanced block design. Defaults to (0.0,) so an existing config
    # keeps its exact meaning. Swept as a lever rather than run as a separate sweep so both arms
    # land in one store and every compute_* table gains an arm dimension for free.
    allocation_mode: Sequence[float] = (0.0,)

    _SCREENING_GRIDS = ("screening_trials", "screening_repeats", "screening_min_reliability",
                        "subjects_noise_lognormal_sigma", "allocation_mode")

    def __post_init__(self):
        super().__post_init__()
        empty = [name for name in self._SCREENING_GRIDS if len(getattr(self, name)) == 0]
        if empty:
            raise ValueError(f"parameter grid(s) must be non-empty: {empty}")
        if any(st < 0 for st in self.screening_trials):
            raise ValueError("`screening_trials` values must be non-negative")
        if any(sr < 0 for sr in self.screening_repeats):
            raise ValueError("`screening_repeats` values must be non-negative")
        if any(s < 0 for s in self.subjects_noise_lognormal_sigma):
            raise ValueError("`subjects_noise_lognormal_sigma` values must be >= 0 "
                             "(0 = use the |t(subjects_noise_df)| family)")
        bad = [mr for mr in self.screening_min_reliability if not (-1 <= mr <= 1)]
        if bad:
            raise ValueError(f"`screening_min_reliability` values must be in [-1, 1], got {bad}")
        bad_alloc = [a for a in self.allocation_mode if a not in (0.0, 1.0)]
        if bad_alloc:
            raise ValueError(
                f"`allocation_mode` values must be 0.0 (random) or 1.0 (designed), got {bad_alloc}"
            )
        # Every (trials, repeats) pair in the Cartesian product is actually simulated, so reject a
        # grid whose product contains a combination with no un-repeated original trial - it would
        # only surface as an assertion deep inside the sweep, hours in.
        infeasible = [(st, sr) for st in self.screening_trials for sr in self.screening_repeats
                      if st > 0 and sr > st - 1]
        if infeasible:
            raise ValueError(
                f"`screening_repeats` must leave at least one distinct trial; infeasible "
                f"(screening_trials, screening_repeats) combinations: {infeasible}"
            )

    def param_grid(self) -> List[TaskV4ExperimentParameters]:
        return [
            TaskV4ExperimentParameters(*p)
            for p in product(
                self.num_subjects,
                self.trials_per_subject,
                self.images_per_trial,
                self.subjects_noise_scale,
                self.subjects_noise_df,
                self.frac_trials_repeated,
                self.perspective_dispersion,
                self.screening_trials,
                self.screening_repeats,
                self.screening_min_reliability,
                self.subjects_noise_lognormal_sigma,
                self.allocation_mode,
            )
        ]


@dataclass
class TaskV5SimulationConfig(TaskV4SimulationConfig):
    """``TaskV4SimulationConfig`` on a **bounded canvas** (see ``canvas`` and ``task_v5_experiment``).

    Adds exactly one lever. ``canvas_softness`` is the exponent of the smooth saturation at the
    canvas walls, swept as a **sensitivity axis** rather than calibrated: unlike aspect and fill it
    has no observable distribution to be drawn from, so the honest treatment is to show the
    conclusions hold across a range instead of picking a value. ``float("inf")`` is exactly hard
    clipping, which the pilot's placement density rules out but which is worth carrying as the
    limiting comparison.

    Aspect and fill are deliberately *not* levers: they describe the apparatus, have measured
    distributions, and are resampled per trial by ``canvas.sample_spec``.

    **Calibration does not transfer from v4.** ``subjects_noise_scale`` is an absolute fraction of
    canvas width here, not a ratio to each trial's arrangement spread, so a v5 sweep needs its own
    calibration run before its numbers mean anything.
    """
    canvas_softness: Sequence[float] = (4.0,)

    def __post_init__(self):
        super().__post_init__()
        if len(self.canvas_softness) == 0:
            raise ValueError("`canvas_softness` must be non-empty")
        if any(sft <= 0 for sft in self.canvas_softness):
            raise ValueError(f"`canvas_softness` values must be positive, got {self.canvas_softness}")

    def param_grid(self) -> List["TaskV5ExperimentParameters"]:
        from SpAM_Simulations.task_v5_experiment import TaskV5ExperimentParameters
        return [
            TaskV5ExperimentParameters(*p)
            for p in product(
                self.num_subjects,
                self.trials_per_subject,
                self.images_per_trial,
                self.subjects_noise_scale,
                self.subjects_noise_df,
                self.frac_trials_repeated,
                self.perspective_dispersion,
                self.screening_trials,
                self.screening_repeats,
                self.screening_min_reliability,
                self.subjects_noise_lognormal_sigma,
                self.allocation_mode,
                self.canvas_softness,
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
