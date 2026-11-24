"""Tests for analysis.s2_semantic_hierarchy.joint_behavior: coverage by image category."""
from __future__ import annotations

import numpy as np

from analysis.s2_semantic_hierarchy import joint_behavior as jb
from analysis.utils.hierarchy import path_parts

B = chr(92)


def test_category_pairs_are_unordered_and_coverage_matrix_is_symmetric():
    categories = np.array(["HO", "HF", "HF", "AB"])
    # condensed order: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    assert list(jb.category_pairs(categories)) == ["HF-HO", "HF-HO", "AB-HO", "HF-HF", "HF-AB", "HF-AB"]
    counts = np.array([1, 0, 2, 3, 0, 1])
    parts = path_parts([f"a{B}b{B}c{B}x{i}.png" for i in range(4)])
    matrix = jb.category_matrix(jb.coverage(counts, parts, categories))
    assert matrix.loc["HF", "HO"] == matrix.loc["HO", "HF"] == 0.5
    assert matrix.loc["HF", "AB"] == matrix.loc["AB", "HF"] == 0.5
    assert matrix.loc["HF", "HF"] == 1.0 and np.isnan(matrix.loc["HB", "HB"])
