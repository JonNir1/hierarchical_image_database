"""Per-subject trial allocation, mirroring SpAM_Task/js/trial_generator.js.

The real task restricts each subject to a random subset of `n_unique` images (derived from
`trials_per_subject`, `images_per_trial`, `frac_images_repeated`), with `n_double` of those
shown in exactly two distinct trials (for within-subject reliability) and the rest in
exactly one. This module ports that allocation algorithm to numpy so the task-v2.3
simulation can reproduce it; it has no notion of noise or distances, only which image index
goes into which trial.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_design_counts(t: int, k: int, r: float) -> Tuple[int, int]:
    """Derive ``(n_unique, n_double)`` from the trial design parameters.

    Mirrors ``trial_generator.js``'s ``n_unique = round(t*k / (1+r))``,
    ``n_double = t*k - n_unique``. ``r`` must stay below 0.5: above that the greedy
    placement in :func:`build_trial_lists` can fail to fill every trial (per the JS comment
    "keep r < 0.5: greedy placement can fail above that").
    """
    assert t > 0, f"`t` (trials_per_subject) must be positive (got {t})"
    assert k > 0, f"`k` (images_per_trial) must be positive (got {k})"
    assert 0 <= r < 0.5, f"`r` (frac_images_repeated) must be in [0, 0.5) (got {r})"
    n_unique = round(t * k / (1 + r))
    n_double = t * k - n_unique
    return n_unique, n_double


def _eligible_trials(trials: List[List[int]], img: int, k: int) -> List[int]:
    """Indices of trials that have room (< k images) and don't already contain `img`."""
    return [i for i, trial in enumerate(trials) if len(trial) < k and img not in trial]


def build_trial_lists(
        active_indices: np.ndarray,
        t: int,
        k: int,
        n_double: int,
        rng: np.random.Generator,
) -> List[np.ndarray]:
    """Allocate `active_indices` into `t` trials of `k` images each.

    Python port of ``buildTrialLists`` (``SpAM_Task/js/trial_generator.js:61-116``): shuffle
    the active set, place the first `n_double` images into exactly 2 distinct trials each
    (within-subject reliability), and the remainder into exactly 1 trial each (the least-full
    eligible one). No image repeats within a single trial.

    :raises RuntimeError: if `(t, k, n_double, len(active_indices))` cannot fill every trial
        (e.g. `frac_images_repeated` too high, or too few active images for the design).
    """
    n_unique = len(active_indices)
    assert 0 <= n_double <= n_unique, f"`n_double` must be between 0 and `n_unique`(={n_unique})"
    assert k > 0, f"`k` (images_per_trial) must be positive (got {k})"
    assert t > 0, f"`t` (trials_per_subject) must be positive (got {t})"

    shuffled = rng.permutation(active_indices)
    double_images = shuffled[:n_double]
    single_images = rng.permutation(shuffled[n_double:])

    trials: List[List[int]] = [[] for _ in range(t)]

    for raw_img in double_images:
        img = int(raw_img)
        eligible = _eligible_trials(trials, img, k)
        if len(eligible) < 2:
            raise RuntimeError(
                f"build_trial_lists: fewer than 2 eligible trials for double-image {img}. "
                "Check trials_per_subject, images_per_trial, and frac_images_repeated."
            )
        chosen = rng.choice(eligible, size=2, replace=False)
        trials[chosen[0]].append(img)
        trials[chosen[1]].append(img)

    for raw_img in single_images:
        img = int(raw_img)
        eligible = _eligible_trials(trials, img, k)
        if not eligible:
            continue  # underfill caught by the validation below
        best = min(eligible, key=lambda i: len(trials[i]))
        trials[best].append(img)

    for i in range(t):
        if len(trials[i]) < k:
            raise RuntimeError(
                f"build_trial_lists: trial {i} has {len(trials[i])} images, expected {k}. "
                "Check design parameters."
            )
        rng.shuffle(trials[i])

    return [np.asarray(trial, dtype=active_indices.dtype) for trial in trials]
