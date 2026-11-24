"""Tests for analysis.s2_semantic_hierarchy.common: the pooled tree bins and NaN-aware agreement."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.s2_semantic_hierarchy import common as c
from analysis.utils.hierarchy import path_parts

B = chr(92)
# Leaves at depth 3, 4 and 5, so pooling (DEEP_TIER = 4) merges the two deepest.
_TOY_PATHS = [
    f"a{B}b{B}c{B}x1.png",
    f"a{B}b{B}c{B}d{B}x2.png",
    f"a{B}b{B}c{B}d{B}e{B}x3.png",
    f"a{B}b{B}c{B}d{B}e{B}x4.png",
]


def test_deep_levels_pool_into_one_tier():
    parts = path_parts(_TOY_PATHS)
    assert list(c.pooled_leaf_depths(parts)) == [3, 4, 4, 4]
    # pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3): LCA depth 3, 3, 3, 4, 4, 5 -> 5 pooled to 4
    assert list(c.lca_bins(parts)) == [3, 3, 3, 4, 4, 4]
    assert list(c.branch_strata(parts)) == ["3x>=4"] * 3 + [">=4x>=4"] * 3


def test_rank_agreement_ignores_unobserved_pairs():
    rng = np.random.default_rng(0)
    a, b = rng.random(200), rng.random(200)
    a_gappy = a.copy()
    a_gappy[::7] = np.nan
    keep = ~np.isnan(a_gappy)
    assert c.rank_agreement(a_gappy, b) == pytest.approx(c.rank_agreement(a[keep], b[keep]))
