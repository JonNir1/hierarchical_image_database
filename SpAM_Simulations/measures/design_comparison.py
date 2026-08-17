"""Stage 2a: what a balanced design buys, measured on the design alone.

No simulated subjects, no MDS, no R. This compares image-to-trial allocation schemes purely as
sampling plans, which is worth doing first because it is cheap, deterministic, and isolates the
allocation effect from every downstream modelling choice. If the arms do not differ here, there is
nothing for the simulation in Stage 2b to find.

Three arms:

* ``random`` - what the deployed task does: one shuffled pool per subject, sliced into consecutive
  trials (``SpAM_Task/js/trial_generator.js:56``), independently per subject.
* ``designed`` - ``block_design.greedy_session_design``, which additionally keeps each subject's
  trials image-disjoint, so it is a drop-in replacement for a real session.
* ``designed_unconstrained`` - the same greedy covering *without* the per-session constraint. Not
  deployable (a subject could see an image twice), and included only to price the constraint.

**Connectivity is not a discriminator here.** The pair graph is a single connected component for
both arms at every N >= 30, so it is reported for completeness rather than as a result. It only
becomes informative at much smaller cohorts.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from SpAM_Simulations.models.block_design import (
    design_stats, greedy_design, greedy_session_design, pair_counts,
)
from SpAM_Simulations.empirical.gt_construction import n_components

ARMS = ("random", "designed", "designed_unconstrained")


def random_sessions(n_images: int, k: int, trials_per_session: int, n_sessions: int,
                    rng: np.random.Generator) -> np.ndarray:
    """The deployed scheme: per subject, shuffle the pool and slice into consecutive trials."""
    need = trials_per_session * k
    return np.stack([
        rng.permutation(n_images)[:need].reshape(trials_per_session, k)
        for _ in range(n_sessions)
    ])


def _unconstrained_sessions(n_images: int, k: int, trials_per_session: int, n_sessions: int,
                            rng: np.random.Generator) -> np.ndarray:
    """A plain greedy covering, chopped into sessions with no disjointness guarantee.

    Prices the per-session constraint. The blocks are taken in the order the greedy emits them, and
    it is regenerated as often as needed to fill ``n_sessions``.
    """
    need = trials_per_session * n_sessions
    blocks: List[np.ndarray] = []
    while len(blocks) < need:
        blocks.extend(greedy_design(n_images, k, 1, rng))
    return np.asarray(blocks[:need], dtype=np.int32).reshape(n_sessions, trials_per_session, k)


def build_sessions(arm: str, n_images: int, k: int, trials_per_session: int, n_sessions: int,
                   rng: np.random.Generator) -> np.ndarray:
    """``(n_sessions, trials_per_session, k)`` allocation for one arm."""
    if arm == "random":
        return random_sessions(n_images, k, trials_per_session, n_sessions, rng)
    if arm == "designed":
        return greedy_session_design(n_images, k, trials_per_session, n_sessions, rng)
    if arm == "designed_unconstrained":
        return _unconstrained_sessions(n_images, k, trials_per_session, n_sessions, rng)
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


def design_report(sessions: np.ndarray, n_images: int) -> Dict[str, float]:
    """Sampling-plan summary for one allocation, including the connectivity prerequisite for MDS."""
    flat = sessions.reshape(-1, sessions.shape[-1])
    counts = pair_counts(flat, n_images)
    weights = (counts > 0).astype(np.float32)
    stats = design_stats(flat, n_images)
    within_session_duplicates = int(sum(
        s.size - np.unique(s).size for s in sessions
    ))
    return {
        **stats,
        "n_sessions": int(sessions.shape[0]),
        "trials_per_session": int(sessions.shape[1]),
        "n_components": n_components(weights),
        "single_component": n_components(weights) == 1,
        # An image seen twice by one subject is not deployable: the task guarantees it cannot happen.
        "within_session_duplicate_images": within_session_duplicates,
    }


def compare_designs(n_images: int = 725, k: int = 20, trials_per_session: int = 18,
                    n_list: Sequence[int] = (30, 50, 75, 300), reps: int = 20, seed: int = 0,
                    arms: Sequence[str] = ARMS, verbose: bool = True) -> pd.DataFrame:
    """Compare allocation arms across cohort sizes. One row per (arm, n_subjects, rep).

    Defaults describe the deployed v4 session: 20 images per trial and 18 distinct trials
    (6 screening + 12 experimental), i.e. 360 distinct images and 18*C(20,2) = 3420 pairs per
    subject, which is 1.303% of the 262,450 pairs in a 725-image set.

    Random coverage therefore follows ~ 1 - exp(-0.01303*N): about 32% at N=30 rising to 98% at
    N=300. That is the yardstick the designed arm has to beat, and it is why the informative window
    is the middle of the range - at N=300 random already covers nearly everything, so there is
    little left to win.
    """
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected some of {ARMS}")
    rows = []
    total = len(arms) * len(n_list) * reps
    with tqdm(total=total, desc="Design comparison", disable=not verbose) as bar:
        for arm in arms:
            for n_subjects in n_list:
                for rep in range(reps):
                    # Distinct stream per cell so arms are compared on independent draws rather
                    # than on the same shuffle.
                    rng = np.random.default_rng([seed, hash(arm) % (2 ** 32), n_subjects, rep])
                    sessions = build_sessions(arm, n_images, k, trials_per_session, n_subjects, rng)
                    rows.append({"arm": arm, "num_subjects": n_subjects, "rep": rep,
                                 **design_report(sessions, n_images)})
                    bar.update(1)
    return pd.DataFrame(rows)


def summarise_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of the headline columns per (arm, num_subjects), for the write-up."""
    cols = ["frac_pairs_covered", "mean_pair_count", "reps_per_image_mean", "reps_per_image_sd",
            "partners_per_image_mean", "wasted_frac", "single_component"]
    present = [c for c in cols if c in df.columns]
    return df.groupby(["arm", "num_subjects"])[present].mean().reset_index()
