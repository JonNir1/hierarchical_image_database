"""Tests for analysis.s2_semantic_hierarchy.mds_dimensionality: the tree-independent choice of d."""
from __future__ import annotations

import pandas as pd

from analysis.s2_semantic_hierarchy import mds_dimensionality as md


def test_both_selection_rules_on_known_curves():
    held_out = pd.DataFrame(
        {"mean": [0.20, 0.25, 0.35, 0.385, 0.395, 0.40], "se": [0.0] * 6},
        index=pd.Index([0, 1, 2, 3, 4, 5], name="ndim"),
    )
    # gain over raw is 0.20; 95% of it is reached first at d=4 (0.395 >= 0.39); the raw row never qualifies
    assert md.choose_high(held_out) == 4
    stress = pd.DataFrame({"stress": [0.60, 0.40, 0.33, 0.32, 0.315, 0.31]}, index=pd.Index(range(1, 7), name="ndim"))
    assert md.choose_low(stress) == 3
