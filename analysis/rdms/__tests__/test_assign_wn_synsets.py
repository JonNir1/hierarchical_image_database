"""
Tests for analysis.rdms.assign_wn_synsets: the rule-based WordNet synset
assignment logic (filename stem + directory -> synset name).

Covers each of the six priority-ordered rules in isolation via small synthetic
paths, the female/male substring-containment edge case in rule 1 ("female"
contains "male" as a substring, so order of the two checks matters), and a
regression test asserting assign_all() reproduces the committed
images/manifest.csv:wn_synset_name column exactly (guards against silent
future drift in STEM_OVERRIDES / STEM_BLACKLIST / rule ordering).
"""
from __future__ import annotations

import pandas as pd

from analysis.rdms.assign_wn_synsets import _parse_path, assign_all, assign_synset


# ---------------------------------------------------------------------------
# _parse_path
# ---------------------------------------------------------------------------

def test_parse_path_strips_trailing_digits_and_lowercases():
    stem, dirname, grandparent = _parse_path(r"Animate\Animal\Bird\Chick12.png")
    assert stem == "chick"
    assert dirname == "bird"
    assert grandparent == "animal"


def test_parse_path_underscore_stem_becomes_space():
    stem, _, _ = _parse_path(r"animate\animal\canine\german_shepherd1.png")
    assert stem == "german shepherd"


# ---------------------------------------------------------------------------
# Rule 1: human face gender token override
# ---------------------------------------------------------------------------

def test_rule1_human_face_female():
    assert assign_synset(r"animate\human\face\asian\female1.png") == "woman.n.01"


def test_rule1_human_face_male():
    assert assign_synset(r"animate\human\face\asian\male1.png") == "man.n.01"


def test_rule1_female_substring_contains_male_edge_case():
    """
    'female' contains 'male' as a substring ('fe' + 'male'), so a naive
    "'male' in raw_stem" check evaluated before the 'female' check would
    misclassify every female-face image as man.n.01. Rule 1 must check
    'female' first.
    """
    assert assign_synset(r"animate\human\face\asian\female3.png") == "woman.n.01"


# ---------------------------------------------------------------------------
# Rule 2: human body man/woman directories
# ---------------------------------------------------------------------------

def test_rule2_human_body_man_dir():
    """Filename ('athlete') is activity/appearance, not identity; dirname wins."""
    assert assign_synset(r"animate\human\body\man\athlete1.png") == "man.n.01"


def test_rule2_human_body_woman_dir():
    assert assign_synset(r"animate\human\body\woman\athlete1.png") == "woman.n.01"


# ---------------------------------------------------------------------------
# Rule 3: manual STEM_OVERRIDES
# ---------------------------------------------------------------------------

def test_rule3_stem_override():
    """'bernese' has no correct standalone WN sense; STEM_OVERRIDES wins over lookup."""
    assert assign_synset(r"animate\animal\canine\bernese1.png") == "bernese_mountain_dog.n.01"


# ---------------------------------------------------------------------------
# Rule 4: STEM_BLACKLIST -- skip stem, fall through to dirname
# ---------------------------------------------------------------------------

def test_rule4_blacklist_stem_falls_to_dirname():
    """'red' is blacklisted as a stem; dirname 'ball' is not colour, so it resolves directly."""
    assert assign_synset(r"inanimate\object\ball\red1.png") == "ball.n.01"


# ---------------------------------------------------------------------------
# Rule 5: standard WN stem lookup, space and underscore forms
# ---------------------------------------------------------------------------

def test_rule5_wn_stem_lookup_space_form():
    assert assign_synset(r"inanimate\natural\food\apple1.png") == "apple.n.01"


def test_rule5_wn_stem_lookup_underscore_form():
    """'german shepherd' has no WN entry as a space-joined lemma; only the underscore form does."""
    assert assign_synset(r"animate\animal\canine\german_shepherd1.png") == "german_shepherd.n.01"


# ---------------------------------------------------------------------------
# Rule 6: dirname fallback, escalating to grandparent for colour dirnames
# ---------------------------------------------------------------------------

def test_rule6_dirname_fallback_colour_to_grandparent():
    """
    Stem has no WN sense at all; dirname 'red' is itself blacklisted (a colour
    subdirectory), so the fallback escalates to the grandparent 'flower'.
    """
    assert assign_synset(r"inanimate\natural\flower\red\zzzznonword1.png") == "flower.n.01"


def test_rule6_dirname_fallback_noncolour_dirname():
    """Stem has no WN sense; dirname is not a colour, so dirname itself is used."""
    assert assign_synset(r"inanimate\object\ball\zzzznonword1.png") == "ball.n.01"


# ---------------------------------------------------------------------------
# Regression: assign_all() reproduces the committed manifest exactly
# ---------------------------------------------------------------------------

def test_assign_all_matches_committed_manifest():
    """
    Guards against silent future drift: any edit to STEM_OVERRIDES,
    STEM_BLACKLIST, or rule ordering that changes a single image's assigned
    synset relative to the currently-committed images/manifest.csv will fail
    this test, forcing an explicit manifest re-generation + review.
    """
    df = pd.read_csv("images/manifest.csv")
    recomputed = assign_all(df)
    mismatches = df[df["wn_synset_name"] != recomputed]
    assert mismatches.empty, (
        f"{len(mismatches)} synset mismatch(es) vs committed manifest:\n"
        f"{mismatches[['curated_path', 'wn_synset_name']].to_string()}"
    )
