"""Task-v5: the v4 model on a **bounded 2-D canvas**.

Identical to task-v4 in every respect except where images can go. v4 (and v3 before it) placed
points on an unbounded plane; the deployed task places them in a rectangle and divides every
distance by that rectangle's diagonal. The omission was not cosmetic - the unbounded model produced
a median per-trial maximum distance of **1.39** on a scale whose ceiling is 1.0, i.e. arrangements
that cannot exist. See :mod:`SpAM_Simulations.canvas` for the geometry and its calibration.

**The canvas is intrinsic here, not a flag.** There is no unbounded branch in v5: a model with a
switch for "obey the geometry of the task" would invite exactly the wrong comparison. v4 remains
available and unchanged for reproducing the published runs, which is why this is a new module rather
than an edit - the same additive pattern v3 -> v4 already used.

**Nothing is duplicated.** ``simulate_task_v4_experiment`` gained a ``trial_simulator`` seam
defaulting to v3's unbounded simulator, so v4 stays bit-for-bit identical (``test_bit_exact_v4``
covers this) and v5 is a thin wrapper that injects the canvas one. Forking 400 lines to change one
step would have undone the property v4's own docstring claims: that the observation model is shared
so the models cannot drift apart.

**Calibrated constants do not carry over.** ``subjects_noise_scale`` changes meaning: on an
unbounded plane the jitter had to be a ratio to each trial's own spread, while on a fixed canvas it
is an absolute fraction of canvas width. Every v3/v4 calibration constant must be re-derived before
a v5 sweep is interpreted, and a v5 run that reuses them is measuring nothing.

**Known gaps, recorded rather than papered over.** The canvas recovers roughly 56% of the empirical
noise-vs-distance turnover (``drop_from_peak`` 0.21 against the pilot's 0.37); the rest needs
ambiguity-dependent noise, which is deliberately **not** modelled here because fitting it would
consume the last observable nothing has been fitted to. And per-trial extent is still too uniform
(max-distance sd 0.089 sampled, against 0.106) because arrangement spread does not depend on trial
content. Both are limitations of the noise model, not of the geometry.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from SpAM_Simulations.canvas import CanvasSpec, DEFAULT_SOFTNESS, make_canvas_trial_simulator
from SpAM_Simulations.task_v4_experiment import (
    TaskV4ExperimentParameters, TaskV4ExperimentResults, simulate_task_v4_experiment,
)

# v5 reuses v4's parameter and result tuples verbatim. The canvas is not a swept experiment
# parameter: it describes the apparatus, not the design, so it is supplied to the simulator rather
# than carried per-cell. `softness` is the exception - it is a sensitivity axis, so a sweep that
# varies it passes a different simulator per arm and records the value in its own column.
TaskV5ExperimentParameters = TaskV4ExperimentParameters
TaskV5ExperimentResults = TaskV4ExperimentResults


def simulate_task_v5_experiment(
        params: TaskV5ExperimentParameters,
        gt_embeddings: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
        allocator=None,
        canvas: Optional[CanvasSpec] = None,
        sample_canvas_per_trial: bool = True,
        softness: float = DEFAULT_SOFTNESS,
) -> tuple:
    """Run one task-v5 experiment: the v4 screening model, on a bounded canvas.

    ``canvas`` pins the geometry; leaving it ``None`` with ``sample_canvas_per_trial=True`` draws
    aspect and fill per trial from the pilot's observed screen shapes, which is the default because
    a fixed spec makes every trial use an identical extent (max-distance sd 0.039 against the
    pilot's 0.106).

    ``softness`` is exposed separately from ``canvas`` because it is the one canvas parameter with
    no observable distribution to sample from, and is therefore swept as a sensitivity axis.
    """
    return simulate_task_v4_experiment(
        params, gt_embeddings, rng, verbose=verbose, allocator=allocator,
        trial_simulator=make_canvas_trial_simulator(
            spec=canvas, sample_per_trial=sample_canvas_per_trial, softness=softness),
    )
