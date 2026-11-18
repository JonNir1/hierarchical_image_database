"""Tests for analysis.utils.hierarchy: tree derivation from curated paths."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform

from analysis.utils.hierarchy import (
    depth_is_ragged, lca_depth_condensed, lca_depth_matrix, path_depths, path_parts,
)

B = chr(92)  # backslash, matching the manifest's Windows-style curated_path

# A deliberately ragged toy tree: leaves at depth 2 and depth 3.
_TOY_PATHS = [
    f"animate{B}bird{B}duck1.png",
    f"animate{B}bird{B}duck2.png",
    f"animate{B}mammal{B}cat{B}cat1.png",
    f"animate{B}mammal{B}cat{B}cat2.png",
    f"animate{B}mammal{B}dog{B}dog1.png",
    f"inanimate{B}tool{B}hammer1.png",
]


def test_path_parts_drops_filename_and_accepts_either_separator():
    parts = path_parts(_TOY_PATHS)
    assert parts[0] == ("animate", "bird") and parts[2] == ("animate", "mammal", "cat")
    assert path_parts(["animate/bird/duck1.png"]) == [("animate", "bird")]
    assert list(path_depths(parts)) == [2, 2, 3, 3, 3, 2]


def test_ragged_detection():
    assert depth_is_ragged(path_parts(_TOY_PATHS)) is True
    assert depth_is_ragged(path_parts([f"a{B}b{B}x1.png", f"a{B}c{B}x2.png"])) is False


def test_lca_depth_is_symmetric_with_zero_diagonal_and_matches_the_tree():
    parts = path_parts(_TOY_PATHS)
    lca = lca_depth_matrix(parts)
    assert np.array_equal(lca, lca.T) and np.all(np.diag(lca) == 0)
    assert lca[0, 1] == 2  # the two ducks share animate/bird
    assert lca[0, 2] == 1  # duck vs cat share only animate
    assert lca[2, 4] == 2  # cat vs dog share animate/mammal
    assert lca[0, 5] == 0  # animate vs inanimate share nothing
    assert np.array_equal(lca_depth_condensed(parts), squareform(lca, checks=False))
