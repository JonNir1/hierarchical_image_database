"""Tests for analysis.build_rdms.hierarchical_clustering on a small ragged toy tree."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.cluster.hierarchy import cophenet, is_valid_linkage
from scipy.spatial.distance import pdist

import analysis.build_rdms.hierarchical_clustering as hcl
from analysis.build_rdms.hierarchy_comparison import lca_depth_condensed
from SpAM_Simulations.measures.cluster_stability import baker_gamma, cophenetic_ranks

# Ragged: leaves at depth 2 and 3, a directory holding images directly plus a subdirectory.
_PARTS = [
    ("animate", "bird"), ("animate", "bird"), ("animate", "bird"),
    ("animate", "mammal", "cat"), ("animate", "mammal", "cat"),
    ("animate", "mammal", "dog"),
    ("animate", "mammal"),
    ("inanimate", "tool"), ("inanimate", "tool"),
    ("inanimate", "food"),
]
_N = len(_PARTS)


def test_linkage_is_valid_and_binary():
    z = hcl.path_tree_to_linkage(_PARTS)
    assert z.shape == (_N - 1, 4)
    assert is_valid_linkage(z)
    assert z[-1, 3] == _N


def test_cophenetic_distance_is_monotone_in_lca_depth():
    z = hcl.path_tree_to_linkage(_PARTS)
    lca = lca_depth_condensed(_PARTS)
    top = max(len(p) for p in _PARTS) + 1
    assert np.allclose(cophenet(z), top - lca)


def test_gt_tree_agrees_perfectly_with_itself():
    z = hcl.path_tree_to_linkage(_PARTS)
    ranks = cophenetic_ranks(z)
    assert baker_gamma(ranks, ranks) == pytest.approx(1.0)


def test_shuffled_leaf_order_gives_same_cophenetic_structure():
    z = hcl.path_tree_to_linkage(_PARTS)
    z_reordered = hcl.path_tree_to_linkage(list(_PARTS))
    assert np.allclose(cophenet(z), cophenet(z_reordered))


def test_km_level_labels_excludes_shallow_images():
    labels = hcl.km_level_labels(_PARTS, 3)
    assert (labels[[0, 1, 2, 6, 7, 8, 9]] == -1).all()
    assert labels[3] == labels[4] != labels[5]


def test_dendrogram_shuffle_test_separates_matching_from_random_structure():
    z = hcl.path_tree_to_linkage(_PARTS)
    matching = hcl.dendrogram_shuffle_test(cophenetic_ranks(z), _PARTS, n_perm=200, seed=0)
    assert matching.observed == pytest.approx(1.0)
    assert matching.p_value < 0.05
    rng = np.random.default_rng(1)
    random_ranks = cophenetic_ranks(hcl.path_tree_to_linkage([_PARTS[j] for j in rng.permutation(_N)]))
    unrelated = hcl.dendrogram_shuffle_test(random_ranks, _PARTS, n_perm=200, seed=2)
    assert unrelated.p_value > 0.05
    assert abs(unrelated.null.mean()) < 0.2


def test_agglomerative_agreement_recovers_planted_embedding():
    rng = np.random.default_rng(3)
    centers = {"animate": np.array([0.0, 0.0]), "inanimate": np.array([10.0, 0.0])}
    coords = np.array([centers[p[0]] + rng.normal(scale=0.3, size=2) for p in _PARTS])
    embedded = {2: pdist(coords)}
    table = hcl.agglomerative_agreement(hcl.path_tree_to_linkage(_PARTS), embedded)
    assert set(table["linkage"]) == {"average", "ward", "complete"}
    assert (table["baker_gamma"] > 0.3).all()
