"""Tests for analysis.s2_semantic_hierarchy.level_separation: global and nested Cohen's d per tree level."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.s2_semantic_hierarchy import common as c
from analysis.s2_semantic_hierarchy import level_separation as ls
from analysis.utils.hierarchy import lca_depth_condensed, path_parts

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
_TOY_N = len(_TOY_PATHS)


@pytest.fixture()
def toy_tree(monkeypatch):
    """Point common.tree_parts at the toy tree instead of the real manifest."""
    parts = path_parts(_TOY_PATHS)
    monkeypatch.setattr(c, "tree_parts", lambda: parts)
    return parts


class TestLevelSeparation:
    def test_between_category_exceeds_within_for_a_tree_distance(self, toy_tree):
        d = 8.0 - lca_depth_condensed(toy_tree)
        out = ls.level_separation(1, {"sem_km": d})
        assert out.loc["sem_km", "between"] > out.loc["sem_km", "within"]
        assert out.loc["sem_km", "cohens_d"] > 0

    def test_constant_rdm_gives_zero_separation(self, toy_tree):
        flat = np.ones(_TOY_N * (_TOY_N - 1) // 2)
        out = ls.level_separation(1, {"flat": flat})
        assert out.loc["flat", "cohens_d"] == pytest.approx(0.0)


class TestNestedContrast:
    def test_nested_excludes_pairs_that_differ_above_the_parent(self, toy_tree):
        # level 2 (mid-level): within = 1, siblings under animate = 2, cross-domain = 100
        lca = lca_depth_condensed(toy_tree)
        d = np.where(lca == 0, 100.0, np.where(lca == 1, 2.0, 1.0))
        glob = ls.level_separation(2, {"x": d})
        nested = ls.level_separation(2, {"x": d}, nested=True)
        assert nested.loc["x", "between"] == pytest.approx(2.0)
        assert glob.loc["x", "between"] > 2.0
        assert nested.loc["x", "n_between"] < glob.loc["x", "n_between"]
