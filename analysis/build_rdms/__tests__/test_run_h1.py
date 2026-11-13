"""Test for analysis.build_rdms.run_h1.family_fdr, the confirmatory H1 multiplicity correction."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import analysis.behavioral.demographics_and_behavior as demographics


def test_family_fdr_is_bh_within_each_cohort_without_leaking_across_cohorts(monkeypatch):
    # run_h1 imports MDS_CACHE_DIR from demographics_and_behavior, but it now lives in
    # cohort_structure; stub it so the import succeeds. Remove once run_h1 is fixed.
    monkeypatch.setattr(demographics, "MDS_CACHE_DIR", Path("."), raising=False)
    family_fdr = importlib.import_module("analysis.build_rdms.run_h1").family_fdr
    permutation = pd.DataFrame({
        "cohort": ["pre", "pre", "post", "post"],
        "statistic": ["kendall_a", "spearman", "kendall_a", "spearman"],  # only tau-a is confirmatory
        "p_value": [0.01, 0.001, 0.30, 0.002],
    })
    gamma_shuffle = pd.DataFrame({
        "cohort": ["pre"] * 3 + ["post"] * 3,
        "linkage": ["average", "ward", "complete"] * 2,
        "p_value": [0.04, 0.03, 0.20, 0.02, 0.50, 0.04],
    })

    result = family_fdr(permutation, gamma_shuffle)

    pooled = multipletests(
        [0.01, 0.04, 0.03, 0.20, 0.30, 0.02, 0.50, 0.04], method="fdr_bh",
    )[1]
    expected = {
        "pre": multipletests([0.01, 0.04, 0.03, 0.20], method="fdr_bh")[1],
        "post": multipletests([0.30, 0.02, 0.50, 0.04], method="fdr_bh")[1],
    }
    for cohort, p_fdr in expected.items():
        family = result[result["cohort"] == cohort]
        assert list(family["test"]) == [
            "tau_a_permutation", "baker_gamma_average", "baker_gamma_ward", "baker_gamma_complete",
        ]
        np.testing.assert_allclose(family["p_fdr"].to_numpy(), p_fdr)
    assert len(result) == 8
    assert not np.allclose(result["p_fdr"].to_numpy(), pooled)  # pooling would give different values
