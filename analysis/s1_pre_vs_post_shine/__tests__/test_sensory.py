"""Tests for analysis.s1_pre_vs_post_shine.sensory: the pre- vs post-SHINE RDM comparison."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from analysis.s1_pre_vs_post_shine import sensory

_N = 40


def _rdm(seed: int) -> np.ndarray:
    return pdist(np.random.default_rng(seed).normal(size=(_N, 4))) + 0.1


def test_identical_rdms_agree_perfectly_in_every_draw():
    d = _rdm(0)
    boot = sensory.bootstrap_rdm_agreement(d, d, n_boot=25)
    assert np.allclose(boot[["spearman", "pearson", "scale_ratio"]].to_numpy(), 1.0)
    # resampled images repeat pairs, and tau-a (unlike tau-b) counts those ties against agreement
    assert (boot["kendall_a"] > 0.99).all()
    assert sensory.rdm_agreement(d, d) == pytest.approx(dict.fromkeys(sensory.STATS, 1.0))


def test_bootstrap_is_seeded_and_its_interval_brackets_the_estimate():
    d_pre = _rdm(0)
    d_post = d_pre + np.random.default_rng(1).normal(scale=0.3, size=d_pre.size)
    first = sensory.bootstrap_rdm_agreement(d_pre, d_post, n_boot=200, seed=3)
    again = sensory.bootstrap_rdm_agreement(d_pre, d_post, n_boot=200, seed=3)
    assert first.equals(again)
    table = sensory.agreement_table(d_pre, d_post, first)
    assert ((table["ci_low"] <= table["estimate"]) & (table["estimate"] <= table["ci_high"])).all()


def test_row_agreement_is_one_for_identical_and_minus_one_for_reversed_distances():
    d = _rdm(0)
    assert np.allclose(sensory.row_agreement(d, d), 1.0)
    assert np.allclose(sensory.row_agreement(d, d.max() - d), -1.0)


def test_shift_between_is_zero_for_a_copy_and_one_for_orthogonal_vectors():
    pre = np.array([[1.0, 0.0], [2.0, 0.0]])
    post = np.array([[1.0, 0.0], [0.0, 3.0]])
    shift = sensory.shift_between(pre, post)
    assert shift["cosine"].to_numpy() == pytest.approx([0.0, 1.0])
    assert shift["euclidean"].to_numpy() == pytest.approx([0.0, np.sqrt(13.0)])
