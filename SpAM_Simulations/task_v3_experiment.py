"""Task-v3 SpAM experiment simulation: a *generative* observation model in coordinate space.

Earlier models (``experiment.py``, ``task_v2_3_experiment.py``, ``task_v2_4_experiment.py``) add
i.i.d. Gaussian noise directly to the ground-truth *distances*. That hides the single biggest gap
between the simulation and the real SpAM task: a subject does not report high-dimensional distances,
they place a trial's ``k`` images on a flat **2-D canvas**. Compressing a ``D``-dimensional
configuration onto 2-D is a lossy, *structured* (non-i.i.d., non-averaging-out) operation, and each
subject weights the underlying dimensions differently ("perspective"). This module models that
directly, working from the ground-truth **coordinates** instead of the distances.

Per-trial observation model (for subject ``s``, a trial of ``k`` images):

1. ``X``  - the trial items' ground-truth coordinates (k x D, PC-aligned by construction of the
   ground-truth generator - see ``simulation.build_ground_truth_embeddings``).
2. ``X' = X * w_s`` - the subject's perspective: a per-PC weight vector ``w_s`` drawn once per
   subject from ``lognormal(0, perspective_dispersion)`` (``dispersion = 0`` -> ``w_s == 1``, i.e.
   everyone shares the ground-truth geometry).
3. ``X'' = X' + N(0, coord_noise)`` - item-level perceptual noise in coordinate space (one draw per
   *coordinate*, not per pair, so a single misplacement perturbs all of an item's pairwise
   distances at once - matching how a real arrangement error works).
4. ``Y = project_2d(X'')`` - a **local, per-trial** classical-MDS / PCA projection onto 2-D. The two
   retained axes are the top-2 variance directions *of this trial's items under this subject's
   weights*, so they rotate from trial to trial and subject to subject. The union of many
   differently-oriented 2-D slices is what lets the aggregate recover ``> 2`` dimensions; a single
   global projection would cap it at rank 2.
5. ``obs = pdist(Y)`` - the recorded 2-D canvas distances. (No per-trial rescaling: like the real
   fixed canvas, every trial shares the same coordinate units, so distances are comparable across
   trials. A fixed-canvas ceiling/saturation is a documented future refinement, not modelled here.)

Design notes:

* **Drops ``frac_images_repeated``.** Task v3.0 (``SpAM_Task/js/trial_generator.js``) shows every
  image in exactly one distinct trial; the only repetition is a verbatim whole-trial repeat via
  ``frac_trials_repeated``. So this module uses ``n_double = 0`` (each subject sees
  ``t_distinct * k`` unique images) and there is no within-subject SNR heuristic (it needed a pair
  observed in two distinct trials, which can no longer happen). The v2.4 **test-retest** diagnostic
  - re-presenting whole trials with fresh noise - is kept.
* The projection method is fixed to classical MDS (PCoA); it is not a swept parameter, so the
  parameter tuple stays fully numeric (the MDS pipeline coerces every parameter to ``float``).
"""
from __future__ import annotations

from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.spatial.distance import num_obs_y, pdist
from scipy.stats import spearmanr
from tqdm import trange

from SpAM_Simulations.design import build_trial_lists, distinct_trial_count, select_repeat_trials
from SpAM_Simulations.experiment import _condensed_pair_indices, _draw_subject_noises
from SpAM_Simulations.helpers import mean_from_sum_and_count

TaskV3ExperimentParameters = NamedTuple("TaskV3ExperimentParameters", [
    ("num_subjects", int),
    ("trials_per_subject", int),
    ("images_per_trial", int),
    ("subjects_noise_scale", float),
    ("subjects_noise_df", int),
    ("frac_trials_repeated", float),
    ("perspective_dispersion", float),
])

TaskV3ExperimentResults = NamedTuple("TaskV3ExperimentResults", [
    ("run_time", datetime),
    ("distances", np.ndarray),
    ("num_obs", np.ndarray),
    ("subject_noises", np.ndarray),
    ("subject_test_retest", np.ndarray),
])


def project_2d(coords: np.ndarray) -> np.ndarray:
    """Classical-MDS / PCA projection of `coords` (k x D) onto its top-2 principal axes (k x 2).

    Centres the points and keeps the two leading principal coordinates (singular vectors scaled by
    their singular values - identical to classical MDS on the Euclidean distances of `coords`). If
    fewer than two non-degenerate directions exist (e.g. ``D == 1``), the missing column(s) are
    zero-filled so the output is always ``(k, 2)``.
    """
    centred = coords - coords.mean(axis=0, keepdims=True)
    # full_matrices=False -> U is (k, r), s is (r,) with r = min(k, D); take the top-2 components.
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    comps = (u * s)[:, :2]
    if comps.shape[1] < 2:  # D == 1 (or a single point): pad to a 2-D canvas
        comps = np.pad(comps, ((0, 0), (0, 2 - comps.shape[1])))
    return comps


def _draw_perspective_weights(D: int, dispersion: float, rng: np.random.Generator) -> np.ndarray:
    """Per-subject PC weight vector ``w_s`` of length ``D`` (a stable per-subject "perspective").

    Drawn from ``lognormal(0, dispersion)`` so weights are positive and multiplicative around 1.
    ``dispersion == 0`` returns all-ones (every subject shares the ground-truth geometry), which
    makes the between-subject signal disagreement vanish and recovers a pure-noise model.
    """
    if dispersion == 0:
        return np.ones(D, dtype=np.float64)
    return np.exp(rng.normal(0.0, dispersion, size=D))


def simulate_task_v3_experiment(
        params: TaskV3ExperimentParameters,
        gt_embeddings: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
        return_per_subject: bool = False,
) -> Tuple[TaskV3ExperimentParameters, TaskV3ExperimentResults]:
    """Simulate multiple subjects under the task-v3 generative (coordinate-space) model.

    Each subject draws their own perspective weights and noise level, sees ``t_distinct * k`` unique
    images allocated into ``t_distinct`` distinct trials, plus ``round(frac_trials_repeated * t)``
    verbatim whole-trial repeats (fresh noise) for test-retest. Returns aggregate condensed mean
    distances (unmeasured pairs NaN), the per-pair observation counts, the per-subject noise levels,
    and the per-subject mean test-retest Spearman (NaN for subjects with no repeats).

    With ``return_per_subject=True`` the return value gains a third element: a ``(num_subjects,
    n_pairs)`` float32 array of each subject's own mean observed distances (NaN where unobserved).
    Used by the pilot calibration to compute between-subject agreement on a simulated cohort with the
    exact same estimator applied to the real subjects (see ``pilot.between_subject_agreement``).
    """
    assert params.num_subjects > 0, f"`num_subjects` must be positive (got {params.num_subjects})"
    assert params.trials_per_subject > 0, f"`trials_per_subject` must be positive (got {params.trials_per_subject})"
    assert params.subjects_noise_scale >= 0, f"`subjects_noise_scale` must be non-negative (got {params.subjects_noise_scale})"
    assert params.subjects_noise_df > 0, f"`subjects_noise_df` must be positive (got {params.subjects_noise_df})"
    assert params.perspective_dispersion >= 0, f"`perspective_dispersion` must be non-negative (got {params.perspective_dispersion})"
    assert 0 <= params.frac_trials_repeated < 1, (
        f"`frac_trials_repeated` must be in [0, 1) (got {params.frac_trials_repeated})"
    )
    gt_embeddings = np.asarray(gt_embeddings, dtype=np.float32)
    assert gt_embeddings.ndim == 2, f"`gt_embeddings` must be a 2-D (N, D) array, got {gt_embeddings.shape}"
    N, D = gt_embeddings.shape

    t_distinct = distinct_trial_count(params.trials_per_subject, params.frac_trials_repeated)
    n_repeats = params.trials_per_subject - t_distinct
    n_unique = t_distinct * params.images_per_trial  # v3: every image in exactly one distinct trial
    assert n_unique <= N, (
        f"`n_unique`(={n_unique} = t_distinct*images_per_trial) exceeds the image pool size (N={N})"
    )

    n_pairs = N * (N - 1) // 2
    all_observations = np.zeros(n_pairs, dtype=np.float64)
    all_n_obs = np.zeros(n_pairs, dtype=np.float64)
    subject_test_retest = np.empty(params.num_subjects, dtype=np.float64)
    per_subject = np.empty((params.num_subjects, n_pairs), dtype=np.float32) if return_per_subject else None
    # Item-level noise lives in coordinate space, so scale it by the coordinate spread (not the
    # distance spread the earlier models used) to keep `subjects_noise_scale` interpretable.
    subject_noises = _draw_subject_noises(
        params.subjects_noise_df,
        params.subjects_noise_scale * float(gt_embeddings.std()),
        params.num_subjects,
        rng
    )
    for s in trange(params.num_subjects, desc="Simulating subjects", disable=not verbose):
        observations, n_obs, test_retest = simulate_task_v3_single_subject(
            subject_noise=subject_noises[s],
            perspective_dispersion=params.perspective_dispersion,
            t_distinct=t_distinct,
            k=params.images_per_trial,
            n_unique=n_unique,
            n_repeats=n_repeats,
            gt_embeddings=gt_embeddings,
            rng=rng,
        )
        all_observations += observations
        all_n_obs += n_obs
        subject_test_retest[s] = test_retest
        if per_subject is not None:
            per_subject[s] = mean_from_sum_and_count(observations, n_obs).astype(np.float32)
    all_observations = np.where(  # unmeasured pairs -> NaN
        all_n_obs > 0, all_observations, np.nan
    )
    results = TaskV3ExperimentResults(
        datetime.now(), all_observations, all_n_obs.astype(np.int16), subject_noises, subject_test_retest
    )
    if return_per_subject:
        return params, results, per_subject
    return params, results


def simulate_task_v3_single_subject(
        subject_noise: float,
        perspective_dispersion: float,
        t_distinct: int,
        k: int,
        n_unique: int,
        n_repeats: int,
        gt_embeddings: np.ndarray,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simulate one subject: ``t_distinct`` distinct trials plus ``n_repeats`` verbatim repeats.

    Returns ``(observations, n_obs, test_retest)``: condensed (1-D) sum/count vectors over all
    completed trials (repeats included, unmeasured pairs 0), and the mean per-repeat Spearman
    correlation between a trial's original and repeat 2-D distance vectors (NaN if ``n_repeats==0``
    or every repeat is degenerate). Perspective weights are drawn once and shared by all of this
    subject's trials (a stable trait); each presentation re-draws its own item-level noise.
    """
    assert subject_noise >= 0, "`subject_noise` must be non-negative"
    assert n_repeats >= 0, "`n_repeats` must be non-negative"
    N, D = gt_embeddings.shape
    assert n_unique <= N, f"`n_unique`(={n_unique}) must not exceed the image pool size (N={N})"

    weights = _draw_perspective_weights(D, perspective_dispersion, rng)
    active_indices = rng.choice(N, size=n_unique, replace=False)
    trials = build_trial_lists(active_indices, t_distinct, k, n_double=0, rng=rng)

    n_pairs = N * (N - 1) // 2
    observations = np.zeros(n_pairs, dtype=np.float64)
    n_obs = np.zeros(n_pairs, dtype=np.float64)
    pair_rows, pair_cols = np.triu_indices(k, k=1)
    distinct_obs: List[np.ndarray] = []  # per-trial 2-D distance vectors, indexed like `trials`
    for trial_images in trials:
        cond_idx, trial_dists = _simulate_trial(
            trial_images, pair_rows, pair_cols, N, gt_embeddings, weights, subject_noise,
            observations, n_obs, rng
        )
        distinct_obs.append(trial_dists)

    # Whole-trial repeats: re-present `n_repeats` trials with fresh noise (same items + weights).
    repeat_idxs = select_repeat_trials(trials, n_repeats, rng)
    retest_corrs: List[float] = []
    for orig_idx in repeat_idxs:
        _, repeat_dists = _simulate_trial(
            trials[orig_idx], pair_rows, pair_cols, N, gt_embeddings, weights, subject_noise,
            observations, n_obs, rng
        )
        retest_corrs.append(_trial_test_retest(distinct_obs[orig_idx], repeat_dists))

    test_retest = float(np.nanmean(retest_corrs)) if retest_corrs else np.nan
    return observations, n_obs, test_retest


def _simulate_trial(
        trial_images: np.ndarray,
        pair_rows: np.ndarray,
        pair_cols: np.ndarray,
        N: int,
        gt_embeddings: np.ndarray,
        weights: np.ndarray,
        subject_noise: float,
        observations: np.ndarray,
        n_obs: np.ndarray,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one trial's 2-D arrangement distances, accumulate them, and return them.

    Reweight the trial items' coordinates by the subject's perspective, add item-level coordinate
    noise, project to a local 2-D arrangement, and record the resulting pairwise distances. Mutates
    `observations`/`n_obs` in place and returns ``(cond_idx, trial_dists)`` (global condensed
    indices and the matching 2-D distance vector, ordered like ``triu_indices(k)`` / ``pdist``).
    """
    coords = gt_embeddings[trial_images] * weights  # (k, D), perspective-weighted
    coords = coords + rng.normal(0.0, subject_noise, size=coords.shape)  # item-level noise
    trial_dists = pdist(project_2d(coords)).astype(np.float32)
    cond_idx = _condensed_pair_indices(trial_images[pair_rows], trial_images[pair_cols], N)
    observations[cond_idx] += trial_dists
    n_obs[cond_idx] += 1
    return cond_idx, trial_dists


def _trial_test_retest(orig: np.ndarray, repeat: np.ndarray) -> float:
    """Spearman correlation between a trial's original and repeat 2-D distance vectors.

    Returns NaN if either vector is constant (Spearman undefined). With noise, two independent
    arrangements of the same items give a finite ``r < 1``; with zero noise the two projections are
    identical, so ``r == 1``.
    """
    if np.ptp(orig) == 0 or np.ptp(repeat) == 0:
        return np.nan
    return float(spearmanr(orig, repeat).statistic)
