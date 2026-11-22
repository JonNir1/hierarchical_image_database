"""Tests for analysis.s1_pre_vs_post_shine.reference_agreement on small synthetic SpAM-like data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

from analysis.s0_build_rdms.behavior.spam_ratings import _combine, _condensed_index
from analysis.s1_pre_vs_post_shine import reference_agreement as ra

_N = 16
_GROUP = np.repeat(np.arange(4), 4)


def _reference() -> np.ndarray:
    within = _GROUP[:, None] == _GROUP[None, :]
    return squareform(np.where(within, 1.0, 3.0) * (1 - np.eye(_N)), checks=False)


def _trials(rng: np.random.Generator, n_participants: int, noise: float) -> ra.TrialsByParticipant:
    ref = squareform(_reference())
    out: ra.TrialsByParticipant = {}
    for p in range(n_participants):
        trials = []
        for _ in range(6):
            members = np.sort(rng.choice(_N, size=6, replace=False))
            i, j = np.triu_indices(len(members), k=1)
            a, b = members[i], members[j]
            trials.append((_condensed_index(a, b, _N), ref[a, b] + rng.normal(scale=noise, size=len(a))))
        out[f"p{p}"] = trials
    return out


def test_cohort_difference_bootstrap_invariants():
    rng = np.random.default_rng(11)
    same_a, same_b, noisy_b = _trials(rng, 12, 0.3), _trials(rng, 12, 0.3), _trials(rng, 12, 6.0)
    ref = _reference()
    kwargs = {"aggregation": "mean", "n_boot": 40, "seed": 1}
    null = ra.bootstrap_cohort_difference(same_a, same_b, {"m": ref}, _N, **kwargs)
    gap = ra.bootstrap_cohort_difference(same_a, noisy_b, {"m": ref}, _N, **kwargs)
    assert abs(null["kendall_a_diff"].mean()) < 0.1 and gap["kendall_a_diff"].mean() > 0.2  # same population vs a real gap

    mask = np.zeros(len(ref), dtype=bool)
    mask[:30] = True
    masked = ra.bootstrap_cohort_difference(
        same_a, same_b, {"m": ref}, _N, aggregation="mean", n_boot=5, seed=2, pair_masks={"own": None, "mask": mask},
    )
    restricted = masked[masked["pairs"] == "mask"]
    assert (restricted[["n_overlap_a", "n_overlap_b"]] <= 30).all().all() and (restricted["n_overlap_a"] > 0).all()

    both = ra.bootstrap_combined_fit_multi(same_a, {"a": ref, "b": ref[::-1].copy()}, _N, aggregation="mean", n_boot=8, seed=3)
    alone = ra.bootstrap_combined_fit_multi(same_a, {"a": ref}, _N, aggregation="mean", n_boot=8, seed=3)
    np.testing.assert_allclose(both.loc[both["model"] == "a", "kendall_a"], alone["kendall_a"])  # shared resamples


def test_loo_fit_has_one_row_per_participant_and_positive_fit():
    fit = ra.loo_model_fit(_trials(np.random.default_rng(4), 12, 0.3), _reference(), _N, aggregation="mean", min_overlap=5)
    assert len(fit) == 12 and fit["model_kendall_a"].mean() > 0.2 and fit["lower_kendall_a"].mean() > 0.2


def test_bh_is_applied_within_each_cohort_never_pooled():
    tests = pd.DataFrame({"cohort": ["pre"] * 3 + ["post"] * 3, "test": list("abcabc"),
                          "p_value": [0.01, 0.04, 0.03, 0.30, 0.02, 0.50]})
    result = ra.bh_within_cohort(tests)
    for cohort in ("pre", "post"):
        family = result[result["cohort"] == cohort]
        np.testing.assert_allclose(family["p_fdr"], multipletests(family["p_value"], method="fdr_bh")[1])
    assert not np.allclose(result["p_fdr"], multipletests(tests["p_value"], method="fdr_bh")[1])
