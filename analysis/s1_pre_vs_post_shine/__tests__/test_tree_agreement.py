"""Tests for analysis.s1_pre_vs_post_shine.tree_agreement on a small ragged toy tree."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.cluster.hierarchy import cophenet, is_valid_linkage
from scipy.spatial.distance import pdist

from analysis.utils.hierarchy import lca_depth_condensed
from analysis.s1_pre_vs_post_shine import tree_agreement as ta
from SpAM_Simulations.measures.cluster_stability import baker_gamma, cophenetic_ranks

_PARTS = [  # ragged: leaves at depth 2 and 3, a directory holding images plus a subdirectory
    ("animate", "bird"), ("animate", "bird"), ("animate", "bird"),
    ("animate", "mammal", "cat"), ("animate", "mammal", "cat"), ("animate", "mammal", "dog"), ("animate", "mammal"),
    ("inanimate", "tool"), ("inanimate", "tool"), ("inanimate", "food"),
]


def test_directory_tree_becomes_an_exact_binary_dendrogram():
    z = ta.path_tree_to_linkage(_PARTS)
    assert is_valid_linkage(z) and z.shape == (len(_PARTS) - 1, 4)
    top = max(len(p) for p in _PARTS) + 1
    np.testing.assert_allclose(cophenet(z), top - lca_depth_condensed(_PARTS))  # shared depth <-> merge height
    np.testing.assert_allclose(cophenet(ta.path_tree_to_linkage(list(_PARTS))), cophenet(z))
    assert baker_gamma(cophenetic_ranks(z), cophenetic_ranks(z)) == pytest.approx(1.0)
    assert (ta.km_level_labels(_PARTS, 3)[[0, 1, 2, 6, 7, 8, 9]] == -1).all()  # shallower images drop out


def test_tree_comparison_separates_matching_from_unrelated_structure_and_recovers_a_planted_embedding():
    rng = np.random.default_rng(1)
    coords = np.vstack([rng.normal(loc=c, scale=0.2, size=(6, 2)) for c in (0.0, 5.0, 10.0)])
    z = ta.perceptual_tree(pdist(coords))
    same = ta.tree_shuffle_test(z, z, n_perm=200, seed=0)
    unrelated = ta.tree_shuffle_test(z, ta.perceptual_tree(pdist(rng.normal(size=(18, 2)))), n_perm=200, seed=1)
    assert same.observed == pytest.approx(1.0) and same.p_value < 0.05 and unrelated.p_value > 0.05

    centres = {"animate": np.array([0.0, 0.0]), "inanimate": np.array([10.0, 0.0])}
    embedded = {3: pdist(np.array([centres[p[0]] + rng.normal(scale=0.3, size=2) for p in _PARTS]))}
    table = ta.agglomerative_agreement(ta.path_tree_to_linkage(_PARTS), embedded)
    assert list(table["linkage"]) == ["average"] and table["baker_gamma"].iloc[0] > 0.3
    shuffled = ta.dendrogram_shuffle_test(cophenetic_ranks(ta.perceptual_tree(embedded[3])), _PARTS, n_perm=100, seed=2)
    assert shuffled.p_value < 0.05
