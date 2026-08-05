"""Tests for analysis.rdms.imagenet_vs_path: path-concept extraction, the 3x3
distance grid, and the argmin bookkeeping. No ResNet forward pass is involved;
classify_top_k() is the only part that needs the model, and it is not exercised
here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis.rdms.imagenet_vs_path as ivp

# ImageNet class indices with stable, well-known synsets.
_COCK = 7        # cock.n.05
_HEN = 8         # hen.n.02
_GOLDFISH = 1    # goldfish.n.01


class TestPathTokens:
    def test_strips_trailing_digits_and_lowercases(self):
        assert ivp.path_tokens(r"animate\animal\body\bird\Duck3.png") == ("duck", "bird")

    def test_keeps_interior_digits(self):
        assert ivp.path_tokens("a/b/mp3player2.png")[0] == "mp3player"

    def test_handles_missing_parent(self):
        assert ivp.path_tokens("duck1.png") == ("duck", "")


class TestPathConceptSynsets:
    def test_three_slots_always_present(self):
        slots = ivp.path_concept_synsets("animate/animal/body/bird/duck3.png")
        assert set(slots) == set(ivp.PATH_SLOTS)

    def test_filename_and_dirname_resolve(self):
        slots = ivp.path_concept_synsets("animate/animal/body/bird/duck3.png")
        assert "duck.n.01" in [s.name() for s in slots["filename"]]
        assert "bird.n.01" in [s.name() for s in slots["dirname"]]

    def test_combined_finds_a_real_compound(self):
        # "tennis ball" is a WordNet lemma; the bare tokens are not enough.
        slots = ivp.path_concept_synsets("inanimate/handmade/ball/tennis1.png")
        assert "tennis_ball.n.01" in [s.name() for s in slots["combined"]]

    def test_combined_tries_both_orderings(self):
        # Reversing the folder/file roles must still reach the same compound.
        slots = ivp.path_concept_synsets("inanimate/handmade/tennis/ball1.png")
        assert "tennis_ball.n.01" in [s.name() for s in slots["combined"]]

    def test_unresolvable_slot_is_empty_not_missing(self):
        slots = ivp.path_concept_synsets("a/zzqqxx/wwvvuu1.png")
        assert slots["filename"] == [] and slots["dirname"] == []


class TestDistanceGrid:
    def test_shape_is_slots_by_ranks(self):
        grid, names = ivp.distance_grid("a/bird/duck1.png", [_COCK, _HEN, _GOLDFISH])
        assert grid.shape == (len(ivp.PATH_SLOTS), 3)
        assert len(names) == len(ivp.PATH_SLOTS) and len(names[0]) == 3

    def test_identical_concept_scores_zero(self):
        # filename "hen" against ImageNet hen must be distance 0.
        grid, _ = ivp.distance_grid("a/bird/hen1.png", [_HEN])
        assert grid[ivp.PATH_SLOTS.index("filename"), 0] == 0.0

    def test_unscorable_cells_are_nan_not_zero(self):
        grid, names = ivp.distance_grid("a/zzqqxx/wwvvuu1.png", [_HEN])
        assert np.all(np.isnan(grid))
        assert names[0][0] is None

    def test_records_which_synset_won(self):
        _, names = ivp.distance_grid("a/bird/hen1.png", [_HEN])
        assert names[ivp.PATH_SLOTS.index("filename")][0] == "hen.n.02"

    def test_polysemous_slot_takes_its_closest_sense(self):
        # "duck" has several senses; the bird sense must win against a bird class.
        grid, names = ivp.distance_grid("a/bird/duck1.png", [_HEN])
        r = ivp.PATH_SLOTS.index("filename")
        assert names[r][0] == "duck.n.01"
        assert grid[r, 0] < ivp.WN_FALLBACK_DIST


class TestSummaries:
    @pytest.fixture()
    def toy(self):
        return pd.DataFrame({
            "curated_path": ["a1.png", "b1.png", "c1.png", "d1.png"],
            "best_slot": ["filename", "filename", "dirname", None],
            "best_rank": [1.0, 3.0, 2.0, np.nan],
            "min_distance": [0.0, 4.0, 2.0, np.nan],
            "d_filename_r1": [0.0, 9.0, 5.0, np.nan],
            "d_filename_r2": [3.0, 7.0, 5.0, np.nan],
            "d_filename_r3": [3.0, 4.0, 5.0, np.nan],
            "d_dirname_r1": [1.0, 9.0, 6.0, np.nan],
            "d_dirname_r2": [1.0, 9.0, 2.0, np.nan],
            "d_dirname_r3": [1.0, 9.0, 6.0, np.nan],
            "d_combined_r1": [np.nan] * 4,
            "d_combined_r2": [np.nan] * 4,
            "d_combined_r3": [np.nan] * 4,
        })

    def test_slot_rank_counts_covers_every_slot(self, toy):
        ct = ivp.slot_rank_counts(toy)
        assert list(ct.index) == list(ivp.PATH_SLOTS)
        assert ct.loc["filename", 1.0] == 1
        assert ct.loc["combined"].sum() == 0

    def test_slot_marginal_distance_ignores_other_slots(self, toy):
        m = ivp.slot_marginal_distance(toy)
        # filename's own best per row is 0, 4, 5 -> one exact match, mean 3.0
        assert m.loc["filename", "n_exact"] == 1
        assert m.loc["filename", "mean"] == pytest.approx(3.0)

    def test_fully_unscorable_slot_reports_zero_scorable(self, toy):
        assert ivp.slot_marginal_distance(toy).loc["combined", "n_scorable"] == 0


class TestDisagreement:
    @pytest.fixture()
    def toy(self):
        return pd.DataFrame({
            "curated_path": [
                r"animate\human\face\caucasian\female1.png",   # differs, human
                r"animate\animal\body\bird\duck1.png",         # agrees
                r"inanimate\handmade\ball\basketball1.png",    # differs, polysemy
                r"inanimate\natural\other\red_leaf1.png",      # unscorable
            ],
            "best_path_synset": ["female.n.01", "duck.n.01", "basketball.n.02", None],
            "manifest_synset": ["woman.n.01", "duck.n.01", "basketball.n.01", "leaf.n.01"],
        })

    def test_detail_lists_only_scorable_disagreements(self, toy):
        d = ivp.disagreement_detail(toy)
        assert len(d) == 2
        assert "duck1.png" not in " ".join(d["curated_path"])       # agreed
        assert "red_leaf1.png" not in " ".join(d["curated_path"])   # unscorable

    def test_detail_flags_polysemy_vs_different_word(self, toy):
        d = ivp.disagreement_detail(toy).set_index("best_path_synset")
        # basketball.n.01/n.02 share the lemma "basketball"; female/woman do not.
        assert bool(d.loc["basketball.n.02", "same_lemma"]) is True
        assert bool(d.loc["female.n.01", "same_lemma"]) is False

    def test_detail_measures_the_gap(self, toy):
        d = ivp.disagreement_detail(toy).set_index("best_path_synset")
        assert d.loc["basketball.n.02", "wn_distance"] > 0
        assert d.loc["female.n.01", "wn_distance"] > 0

    def test_detail_marks_human_images(self, toy):
        d = ivp.disagreement_detail(toy).set_index("best_path_synset")
        assert bool(d.loc["female.n.01", "is_human"]) is True
        assert bool(d.loc["basketball.n.02", "is_human"]) is False

    def test_summary_by_human_separates_the_groups(self, toy):
        s = ivp.disagreement_summary(toy, by="human")
        assert s.loc["human", "n_agree"] == 0
        assert s.loc["non-human", "n_agree"] == 1

    def test_summary_excludes_unscorable_from_the_rate(self, toy):
        s = ivp.disagreement_summary(toy, by="human")
        # red_leaf counts toward n but not n_scorable, so it cannot depress the rate
        assert s.loc["non-human", "n"] == 3
        assert s.loc["non-human", "n_scorable"] == 2
        assert s.loc["non-human", "agree_pct"] == pytest.approx(50.0)

    def test_summary_by_branch_groups_on_two_levels(self, toy):
        s = ivp.disagreement_summary(toy, by="branch")
        assert set(s.index) == {"animate/human", "animate/animal",
                                "inanimate/handmade", "inanimate/natural"}

    def test_summary_rejects_unknown_grouping(self, toy):
        with pytest.raises(ValueError, match="must be"):
            ivp.disagreement_summary(toy, by="nonsense")
