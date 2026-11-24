"""Tests for analysis.s2_semantic_hierarchy.tree_recovery: flat cuts of an RDM's own dendrogram, and HDBSCAN."""
from __future__ import annotations

from analysis.s2_semantic_hierarchy import tree_recovery as tr
from analysis.utils.hierarchy import lca_depth_condensed, path_parts

B = chr(92)
_TOY_PATHS = [  # 2 domains x 2 categories x 2 basic levels x 2 images, all leaves at depth 3
    f"d{a}{B}c{b}{B}k{k}{B}i{i}.png" for a in range(2) for b in range(2) for k in range(2) for i in range(2)
]


def test_sem_km_dendrogram_cut_at_true_category_counts_recovers_every_level():
    parts = path_parts(_TOY_PATHS)
    km = 2.0 * (3 - lca_depth_condensed(parts))
    table = tr.flat_cut_ari(tr.rdm_trees({"sem_km": km}), parts)
    assert list(table["n_categories"]) == [2, 4, 8]
    assert list(table["n_clusters"]) == [2, 4, 8]
    assert (table["ari"] == 1.0).all()
    assert (table["robust_ari"] == 1.0).all()


def test_hdbscan_recovers_two_separated_domains_without_noise():
    parts = path_parts(_TOY_PATHS)
    lca = lca_depth_condensed(parts)
    dist = 0.1 * (3 - lca) + 10.0 * (lca == 0)  # two tight domain groups far apart
    table = tr.hdbscan_table({"sem_km": dist}, parts, levels=[1], min_cluster_sizes=(5,))
    row = table.iloc[0]
    assert (row["n_clusters"], row["frac_noise"], row["ari"]) == (2, 0.0, 1.0)
    assert row["expected_role"] == "ceiling"
