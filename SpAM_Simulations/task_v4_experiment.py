"""Task-v4 SpAM experiment simulation: task-v3's generative model plus a **screening block**.

The observation model is unchanged from ``task_v3_experiment`` - per-subject perspective weights,
a local per-trial 2-D projection, canvas placement noise - and its building blocks are imported
from there rather than duplicated, so the two models cannot drift apart. What v4 adds is the
deployed SpAM_Task v4.0 **screening stage**: a short preliminary block that every candidate
completes, after which unreliable candidates are excluded and replaced.

Why this belongs in the simulation. ``_draw_subject_noises`` (see ``experiment.py``) gives each
subject a placement precision drawn from ``|t(df)|``, a distribution with a heavy right tail - a
minority of very unreliable subjects. Screening truncates exactly that tail, so its effect on
required-N is a first-class simulable question rather than an assumption: recruiting *better*
subjects and recruiting *more* subjects are competing uses of the same budget, and the earlier
sweeps established that reliability is the dominant lever (required-N is proportional to
``(1 - R) / R``).

Screening model (mirrors ``SpAM_Task/js/utils.js::evaluateScreening``):

* Each candidate completes ``screening_trials`` main-trial slots, ``screening_repeats`` of which
  are verbatim whole-trial repeats, yielding one test-retest Spearman per repeat.
* The candidate is **retained** iff the *minimum* of those per-repeat correlations is at least
  ``screening_min_reliability``. This is the per-repeat minimum rule the deployed task uses, not
  an aggregate: a single bad repeat excludes.
* Candidates are drawn until ``num_subjects`` are retained, mirroring the real recruit-until-N
  stopping rule ("excluded subjects are replaced"), so the ``num_subjects`` axis keeps meaning
  "analysed cohort size" and stays comparable to the v3 sweeps.
* Each candidate gets ONE image pool, partitioned across the screening and main stages so no image
  appears in both (``trial_generator.js::partitionIntoStages``). ``trials_per_subject`` therefore
  counts the **main stage only** - at the deployed design that is 14 (12 distinct + 2 repeats), on
  top of which the screening block adds 8, for 22 trials and 360 distinct images per subject.
* A retained subject's screening trials **are data** - the screening stage uses the same stimuli
  and the same task, and the pre-registered analysis includes it - so their observations are
  accumulated. Screened-out candidates contribute nothing.

``screening_min_reliability = -1.0`` disables exclusion while still running the block (a Spearman
correlation is always >= -1), which isolates the effect of *exclusion* while holding the number of
collected trials fixed across arms. ``screening_trials = 0`` skips the block entirely and
reproduces the task-v3 model exactly.

**Known limitation.** The generative model has no notion of ``num_moves`` or of a trial's
arrangement spread, so only the reliability criterion is simulable here. The deployed task also
screens on move-ratio and distance-SD fail rates. Results are therefore an upper bound on what
reliability-based screening alone can buy.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from tqdm import trange

from SpAM_Simulations.design import build_trial_lists, distinct_trial_count, select_repeat_trials
from SpAM_Simulations.experiment import _draw_subject_noises
from SpAM_Simulations.helpers import mean_from_sum_and_count
from SpAM_Simulations.task_v3_experiment import (
    _draw_perspective_weights, _simulate_trial, _trial_test_retest, _trial_test_retest_procrustes,
)

TaskV4ExperimentParameters = NamedTuple("TaskV4ExperimentParameters", [
    ("num_subjects", int),
    ("trials_per_subject", int),
    ("images_per_trial", int),
    ("subjects_noise_scale", float),
    ("subjects_noise_df", int),
    ("frac_trials_repeated", float),
    ("perspective_dispersion", float),
    ("screening_trials", int),            # screening-stage main-trial slots (0 = no screening)
    ("screening_repeats", int),           # of which verbatim repeats (test-retest probes)
    ("screening_min_reliability", float), # exclude if min per-repeat rho < this (-1 = exclude nobody)
])

SubjectRun = NamedTuple("SubjectRun", [
    ("observations", np.ndarray),          # condensed sum of this stage's observed distances
    ("n_obs", np.ndarray),                 # condensed observation counts
    ("repeat_correlations", List[float]),  # one test-retest Spearman per repeat (NaN if degenerate)
    ("repeat_procrustes", List[float]),    # one Procrustes M^2 per repeat (NaN if degenerate)
])

TaskV4ExperimentResults = NamedTuple("TaskV4ExperimentResults", [
    ("run_time", datetime),
    ("distances", np.ndarray),
    ("num_obs", np.ndarray),
    ("subject_noises", np.ndarray),                      # RETAINED subjects' placement precisions
    ("subject_test_retest", np.ndarray),                 # Spearman(orig, repeat) 2-D distances; higher=better
    ("subject_test_retest_procrustes", np.ndarray),      # Procrustes M^2 of the 2-D arrangements; LOWER=better
    ("n_candidates_screened", int),                      # candidates simulated to retain num_subjects
    ("screening_pass_rate", float),                      # num_subjects / n_candidates_screened
])


def simulate_task_v4_experiment(
        params: TaskV4ExperimentParameters,
        gt_embeddings: np.ndarray,
        rng: np.random.Generator,
        verbose: bool = True,
        return_per_subject: bool = False,
) -> Tuple[TaskV4ExperimentParameters, TaskV4ExperimentResults]:
    """Simulate a screened cohort under the task-v4 model.

    Candidates are drawn until ``num_subjects`` pass screening; each retained subject then
    completes the main stage. Returns aggregate condensed mean distances (unmeasured pairs NaN),
    per-pair observation counts, and per-retained-subject noise/test-retest diagnostics, plus the
    recruitment cost (``n_candidates_screened``, ``screening_pass_rate``).

    ``subject_noises`` and the two test-retest arrays describe the **retained** cohort only -
    that is the population the study actually analyses, and the shift in their distribution
    relative to the unscreened draw is precisely what screening is meant to buy.

    With ``return_per_subject=True`` the return value gains a third element: a ``(num_subjects,
    n_pairs)`` float32 array of each retained subject's own mean observed distances (NaN where
    unobserved), matching ``simulate_task_v3_experiment``'s contract.
    """
    _validate(params)
    gt_embeddings = np.asarray(gt_embeddings, dtype=np.float32)
    assert gt_embeddings.ndim == 2, f"`gt_embeddings` must be a 2-D (N, D) array, got {gt_embeddings.shape}"
    N, _ = gt_embeddings.shape

    k = params.images_per_trial
    t_distinct = distinct_trial_count(params.trials_per_subject, params.frac_trials_repeated)
    n_repeats = params.trials_per_subject - t_distinct
    n_unique = t_distinct * k  # v3/v4: every image in exactly one distinct trial
    assert n_unique <= N, (
        f"`n_unique`(={n_unique} = t_distinct*images_per_trial) exceeds the image pool size (N={N})"
    )
    screen_distinct = params.screening_trials - params.screening_repeats
    screen_unique = screen_distinct * k
    # Both stages draw from ONE disjoint per-subject pool (see the candidate loop), so the pool
    # must accommodate their sum, not each separately.
    assert screen_unique + n_unique <= N or params.screening_trials == 0, (
        f"screening block needs {screen_unique} unique images on top of the main stage's "
        f"{n_unique}, exceeding the pool size (N={N})"
    )

    n_pairs = N * (N - 1) // 2
    all_observations = np.zeros(n_pairs, dtype=np.float64)
    all_n_obs = np.zeros(n_pairs, dtype=np.float64)
    subject_noises = np.empty(params.num_subjects, dtype=np.float64)
    subject_test_retest = np.empty(params.num_subjects, dtype=np.float64)
    subject_test_retest_procrustes = np.empty(params.num_subjects, dtype=np.float64)
    per_subject = np.empty((params.num_subjects, n_pairs), dtype=np.float32) if return_per_subject else None

    noise_pool = _CandidateNoisePool(
        params.subjects_noise_df, params.subjects_noise_scale, params.num_subjects, rng
    )
    retained = 0
    n_candidates = 0
    with trange(params.num_subjects, desc="Simulating subjects", disable=not verbose) as bar:
        while retained < params.num_subjects:
            n_candidates += 1
            noise = noise_pool.next()

            screening = None
            screen_images, main_images = None, None
            if params.screening_trials > 0:
                # ONE pool per candidate, partitioned across the two stages, so no image appears in
                # both - matching `trial_generator.js::partitionIntoStages`. Drawing the two stages
                # independently would overlap them (at the deployed design, ~40 of a subject's 360
                # images), manufacturing within-subject cross-stage pair observations the real task
                # cannot produce.
                pool = rng.choice(N, size=screen_unique + n_unique, replace=False)
                screen_images, main_images = pool[:screen_unique], pool[screen_unique:]
                screening = simulate_task_v4_single_subject(
                    subject_noise=noise, perspective_dispersion=params.perspective_dispersion,
                    t_distinct=screen_distinct, k=k, n_unique=screen_unique,
                    n_repeats=params.screening_repeats, gt_embeddings=gt_embeddings, rng=rng,
                    image_indices=screen_images,
                )
                if not _passes_screening(screening.repeat_correlations,
                                         params.screening_min_reliability):
                    continue  # screened out: their data is discarded and they are replaced

            main = simulate_task_v4_single_subject(
                subject_noise=noise, perspective_dispersion=params.perspective_dispersion,
                t_distinct=t_distinct, k=k, n_unique=n_unique, n_repeats=n_repeats,
                gt_embeddings=gt_embeddings, rng=rng, image_indices=main_images,
            )
            # A retained subject's screening trials are analysed data, so pool both stages.
            observations = main.observations
            n_obs = main.n_obs
            corrs = list(main.repeat_correlations)
            m2s = list(main.repeat_procrustes)
            if screening is not None:
                observations = observations + screening.observations
                n_obs = n_obs + screening.n_obs
                corrs += list(screening.repeat_correlations)
                m2s += list(screening.repeat_procrustes)

            all_observations += observations
            all_n_obs += n_obs
            subject_noises[retained] = noise
            subject_test_retest[retained] = _nanmean_or_nan(corrs)
            subject_test_retest_procrustes[retained] = _nanmean_or_nan(m2s)
            if per_subject is not None:
                per_subject[retained] = mean_from_sum_and_count(observations, n_obs).astype(np.float32)
            retained += 1
            bar.update(1)

    all_observations = np.where(all_n_obs > 0, all_observations, np.nan)  # unmeasured pairs -> NaN
    results = TaskV4ExperimentResults(
        datetime.now(), all_observations, all_n_obs.astype(np.int16), subject_noises,
        subject_test_retest, subject_test_retest_procrustes,
        n_candidates, params.num_subjects / n_candidates,
    )
    if return_per_subject:
        return params, results, per_subject
    return params, results


class _CandidateNoisePool:
    """Supplies per-candidate placement precisions, one at a time, from ``_draw_subject_noises``.

    That function normalises a *batch* so its sample mean is exactly ``subjects_noise_scale``,
    which is what gives ``subjects_noise_scale`` its meaning. Two consequences shape this class:

    * **Drawing one at a time is wrong.** A batch of size 1 divides by its own single value, so
      every candidate would come back as exactly ``subjects_noise_scale`` and the heavy ``|t(df)|``
      tail - the very heterogeneity screening selects against - would vanish.
    * **Screening must not renormalise.** The batch is the *candidate* pool, not the retained
      cohort; the retained subjects are a selected subsample whose mean noise falls **below**
      ``subjects_noise_scale``. That shift is exactly what screening buys, so it must survive.

    Batches are therefore drawn ``num_subjects`` at a time and consumed in order, refilling when
    exhausted (the number of candidates is not known in advance). With screening disabled exactly
    one batch is drawn and consumed in order, so the RNG stream matches task-v3's and the two
    models agree bit-for-bit.
    """

    def __init__(self, df: int, scale: float, batch_size: int, rng: np.random.Generator):
        self._df, self._scale, self._batch_size, self._rng = df, scale, max(batch_size, 1), rng
        self._buffer: np.ndarray = np.empty(0, dtype=np.float64)
        self._pos = 0

    def next(self) -> float:
        if self._pos >= self._buffer.size:
            self._buffer = _draw_subject_noises(self._df, self._scale, self._batch_size, self._rng)
            self._pos = 0
        value = float(self._buffer[self._pos])
        self._pos += 1
        return value


def simulate_task_v4_single_subject(
        subject_noise: float,
        perspective_dispersion: float,
        t_distinct: int,
        k: int,
        n_unique: int,
        n_repeats: int,
        gt_embeddings: np.ndarray,
        rng: np.random.Generator,
        image_indices: Optional[np.ndarray] = None,
) -> SubjectRun:
    """Simulate one stage for one subject: ``t_distinct`` distinct trials plus ``n_repeats`` repeats.

    Identical in mechanics to ``task_v3_experiment.simulate_task_v3_single_subject`` - and it
    delegates to that module's ``_simulate_trial`` so the observation model is literally shared -
    but returns the **per-repeat** reliability values instead of their mean. Task-v4 needs them
    unaggregated for two reasons: the screening rule is a minimum over repeats (not a mean), and a
    retained subject's screening and main repeats are pooled before averaging, which is not
    recoverable from two separate means.

    Perspective weights are drawn once here, so calling this twice for the same subject (screening
    then main stage) gives them two independent perspectives. That is deliberate: the alternative -
    threading one weight vector through both - would make the screening block's reliability
    partially predictive of the main stage's *perspective* as well as its precision, which is not
    what the deployed screening measures. Placement precision, the trait screening actually selects
    on, IS shared: both calls receive the same ``subject_noise``.

    ``image_indices`` supplies this stage's image pool explicitly instead of drawing it here. The
    caller uses it to partition ONE per-subject pool across the screening and main stages, so no
    image appears in both - mirroring ``trial_generator.js::partitionIntoStages``. Left as ``None``
    the pool is drawn here, which is what task-v3 does and keeps the no-screening arm bit-exact.
    """
    assert subject_noise >= 0, "`subject_noise` must be non-negative"
    assert n_repeats >= 0, "`n_repeats` must be non-negative"
    N, D = gt_embeddings.shape
    assert n_unique <= N, f"`n_unique`(={n_unique}) must not exceed the image pool size (N={N})"

    weights = _draw_perspective_weights(D, perspective_dispersion, rng)
    if image_indices is None:
        active_indices = rng.choice(N, size=n_unique, replace=False)
    else:
        active_indices = np.asarray(image_indices)
        assert active_indices.size == n_unique, (
            f"`image_indices` has {active_indices.size} entries, expected n_unique={n_unique}"
        )
    trials = build_trial_lists(active_indices, t_distinct, k, n_double=0, rng=rng)

    n_pairs = N * (N - 1) // 2
    observations = np.zeros(n_pairs, dtype=np.float64)
    n_obs = np.zeros(n_pairs, dtype=np.float64)
    pair_rows, pair_cols = np.triu_indices(k, k=1)
    distinct_obs: List[np.ndarray] = []   # per-trial 2-D distance vectors, indexed like `trials`
    distinct_arr: List[np.ndarray] = []   # per-trial 2-D arrangements (k, 2), indexed like `trials`
    for trial_images in trials:
        _, trial_dists, arrangement = _simulate_trial(
            trial_images, pair_rows, pair_cols, N, gt_embeddings, weights, subject_noise,
            observations, n_obs, rng
        )
        distinct_obs.append(trial_dists)
        distinct_arr.append(arrangement)

    # Whole-trial repeats: re-present `n_repeats` trials with fresh noise (same items + weights).
    repeat_idxs = select_repeat_trials(trials, n_repeats, rng)
    corrs: List[float] = []
    m2s: List[float] = []
    for orig_idx in repeat_idxs:
        _, repeat_dists, repeat_arr = _simulate_trial(
            trials[orig_idx], pair_rows, pair_cols, N, gt_embeddings, weights, subject_noise,
            observations, n_obs, rng
        )
        corrs.append(_trial_test_retest(distinct_obs[orig_idx], repeat_dists))
        m2s.append(_trial_test_retest_procrustes(distinct_arr[orig_idx], repeat_arr))
    return SubjectRun(observations, n_obs, corrs, m2s)


def _passes_screening(repeat_correlations: List[float], min_reliability: float) -> bool:
    """Apply the deployed per-repeat **minimum** rule (``SpAM_Task/js/utils.js::evaluateScreening``).

    A candidate is excluded if *any* single repeat's test-retest Spearman falls below the
    threshold - not if their average does. NaN repeats (a degenerate arrangement, so the
    correlation is undefined) are skipped rather than treated as failures, matching the deployed
    task's behaviour of skipping the criterion when no usable repeat exists; a candidate with no
    usable repeat at all therefore passes, since there is no evidence against them.
    """
    usable = [c for c in repeat_correlations if not np.isnan(c)]
    if not usable:
        return True
    return min(usable) >= min_reliability


def _nanmean_or_nan(values: List[float]) -> float:
    """Mean of the non-NaN entries, or NaN when there are none (no all-NaN RuntimeWarning)."""
    usable = [v for v in values if not np.isnan(v)]
    return float(np.mean(usable)) if usable else np.nan


def _validate(params: TaskV4ExperimentParameters) -> None:
    assert params.num_subjects > 0, f"`num_subjects` must be positive (got {params.num_subjects})"
    assert params.trials_per_subject > 0, f"`trials_per_subject` must be positive (got {params.trials_per_subject})"
    assert params.images_per_trial > 0, f"`images_per_trial` must be positive (got {params.images_per_trial})"
    assert params.subjects_noise_scale >= 0, f"`subjects_noise_scale` must be non-negative (got {params.subjects_noise_scale})"
    assert params.subjects_noise_df > 0, f"`subjects_noise_df` must be positive (got {params.subjects_noise_df})"
    assert params.perspective_dispersion >= 0, f"`perspective_dispersion` must be non-negative (got {params.perspective_dispersion})"
    assert 0 <= params.frac_trials_repeated < 1, (
        f"`frac_trials_repeated` must be in [0, 1) (got {params.frac_trials_repeated})"
    )
    assert params.screening_trials >= 0, f"`screening_trials` must be non-negative (got {params.screening_trials})"
    assert 0 <= params.screening_repeats <= max(params.screening_trials - 1, 0), (
        f"`screening_repeats` must be in [0, screening_trials - 1] "
        f"(got {params.screening_repeats} with screening_trials={params.screening_trials})"
    )
    assert -1 <= params.screening_min_reliability <= 1, (
        f"`screening_min_reliability` must be in [-1, 1] (got {params.screening_min_reliability})"
    )
