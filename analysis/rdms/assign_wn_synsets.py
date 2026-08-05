"""
Assign WordNet synsets to each image from its filename stem and directory label.

Reads images/manifest.csv, derives a WordNet noun synset for each image, and
writes the result back as the wn_synset_name column (creating it if absent or
overwriting if present).

THIS IS THE REFERENCE CONCEPT ASSIGNMENT FOR THE DATASET.
`images/manifest.csv:wn_synset_name` is the single source of truth for "what
concept is this image", and D_sem_wn is built from it (semantic_wn_dir.py).
Future analyses needing a per-image concept should read that column rather than
re-deriving one from the path or from a classifier. Reasons, measured in
analysis/rdms/imagenet_vs_path.py over all 725 images:

  - Automatic path parsing reaches a different synset than this module's
    assignment for 376 of 717 resolvable images, and none of those differences
    are ties: the median gap is 9 WordNet edges.
  - It fails hardest exactly where the dataset is most distinctive. Agreement is
    84% within animate/animal but 3.7% within animate/human, because a bare
    lemma cannot get there: WordNet offers no `woman` sense for the word
    "female", so `female1.png` resolves to female.n.01 (any female organism)
    where the curated concept is woman.n.01.
  - Polysemy silently picks the wrong sense elsewhere too, sometimes very far
    off: basketball.n.02 (the game) against basketball.n.01 (the ball) are 18
    edges apart; likewise hand.n.08 (a hired worker) against hand.n.01 (the body
    part) at 14, and banana.n.02 (the plant) against banana.n.01 (the fruit).
  - Eight images have multi-word filenames that are not WordNet lemmas at all
    (skunk_pig, fishing_float, red_leaf, ...) and are unresolvable by any
    automatic rule, yet each has an obvious curated concept.
  - The ResNet-50/ImageNet route was tried first and abandoned: ImageNet-1K has
    no person, face, or human-body classes, so all 164 human images are
    unclassifiable in principle, and top-1 was the best of the top-3 for only
    58% of images even where it did apply.

Two limits worth stating alongside that. First, this assignment fixes the
*concepts*, not the metric: WordNet shortest-path still measures taxonomic
bookkeeping, which is why baboon-capuchin scores 18 edges while baboon-macaque
scores 2. Second, because these labels come from the same curation that built
the directory tree, D_sem_wn and D_sem_km are not independent measurements of
semantics; they share a source, and their observed rho = 0.29 should be read
with that in mind.

Assignment rules (evaluated in order):

1. Human-face images (path contains both 'human' and 'face'):
     'female' anywhere in raw filename stem -> woman.n.01
     'male'   anywhere in raw filename stem -> man.n.01
     (labelled by photographer's ethnicity + gender tokens, not a concept name)

2. Human-body man/woman dirs (path contains 'human\\body\\man' or 'human\\body\\woman'):
     dirname == 'man'   -> man.n.01
     dirname == 'woman' -> woman.n.01
     (filenames encode activity/appearance, not identity; dirname is the concept)

3. Stem in STEM_OVERRIDES -> use the mapped synset directly.
   Handles polysemous stems where WN's most-frequent sense is wrong, and
   multi-word stems that require an underscore-form not in WN's lemma list.

4. Stem in STEM_BLACKLIST -> skip stem, fall through to dirname/grandparent.
   Colour words used as car filenames (blue1.png, red1.png, …) and other
   generic stems that would map to useless synsets.

5. WN stem lookup: try both the space form and the underscore form.
   e.g. 'german shepherd' / 'german_shepherd' -> german_shepherd.n.01

6. Dirname fallback.
   If dirname is a colour (member of STEM_BLACKLIST), use the GRANDPARENT
   directory name instead (handles flower/red, flower/blue, … subdirs).

stem    = filename without extension, trailing digits stripped, lowercased,
          underscores replaced with spaces.
dirname = immediate parent directory name, lowercased.

Usage (from repo root, with .venv active):
    python -m analysis.rdms.assign_wn_synsets

Output: images/manifest.csv updated in place (wn_synset_name column).
"""
from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath

import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError

MANIFEST_PATH = Path(__file__).parent.parent.parent / "images" / "manifest.csv"


# ---------------------------------------------------------------------------
# Override table — polysemous and multi-word stems
# ---------------------------------------------------------------------------

STEM_OVERRIDES: dict[str, str] = {
    # Animals / breeds
    "bernese":              "bernese_mountain_dog.n.01",
    "skunk pig":            "peccary.n.01",
    "sugar glider":         "flying_phalanger.n.01",
    "grey crowned crane":   "crane.n.05",
    "cat angora":           "angora.n.04",     # angora.n.01 is the city Ankara
    # Sports / games
    "billiard":             "billiard_ball.n.01",
    "bedminton racket":     "badminton_racket.n.01",
    "fishing float":        "bob.n.05",        # bob.n.01 is a haircut
    "fishing hook":         "fishhook.n.01",
    "ping pong":            "table_tennis.n.01",
    "tennis":               "tennis_ball.n.01",  # tennis.n.01 is the sport
    # Tools / instruments
    "fire bellow":          "bellows.n.01",
    "garden fork":          "pitchfork.n.01",
    "ink pad":              "pad.n.03",
    "screw":                "screw.n.04",      # screw.n.01 is prison_guard (slang)
    "hose":                 "hose.n.03",       # hose.n.01 is hosiery
    # Kitchen
    "muffin pan":           "kitchen_utensil.n.01",
    "pizza knife":          "kitchen_utensil.n.01",
    # Vehicles
    "fighter jet":          "jet.n.01",
    "plane":                "airplane.n.01",
    "sailing ship":         "sailing_vessel.n.01",
    "police":               "police_car.n.01",
    # Electronics
    "cd":                   "compact_disk.n.01",   # cd.n.01 is cadmium
    # Plants / food
    "sabra":                "prickly_pear.n.01",   # sabra.n.01 is an Israeli person
    "moth orcid":           "orchid.n.01",
    "woodsorrel":           "oxalis.n.01",
    "pea stock":            "pea.n.01",
    "pepper green":         "capsicum.n.02",    # pepper.n.01/03 is black pepper
    "pepper hot":           "capsicum.n.02",
    "pepper red":           "capsicum.n.02",
    "red leaf":             "leaf.n.01",
    "lettuce":              "lettuce.n.02",     # lettuce.n.01 is boodle (money slang)
    "jackberry":            "produce.n.01",
}

# Stems (and dirnames) that should skip to the next fallback level.
# Primarily colour words used as car filenames (blue1.png, red1.png, …) and
# colour subdirectories in the flower category.
STEM_BLACKLIST: frozenset[str] = frozenset({
    "blue", "red", "green", "dark", "light", "pink", "white", "yellow",
    "other", "gummy", "female", "male",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_path(curated_path: str) -> tuple[str, str, str]:
    """Return (stem, dirname, grandparent_dir) for a manifest curated_path."""
    p = PureWindowsPath(curated_path)
    stem = re.sub(r"\d+$", "", p.stem).lower().replace("_", " ").strip()
    dirname = p.parts[-2].lower() if len(p.parts) >= 2 else ""
    grandparent = p.parts[-3].lower() if len(p.parts) >= 3 else ""
    return stem, dirname, grandparent


def _lookup(word: str) -> wn.Synset | None:
    """
    First noun synset for `word`, trying both space and underscore forms.
    Returns None if no synset found or on WordNetError.
    """
    for form in (word, word.replace(" ", "_")):
        if not form:
            continue
        try:
            syns = wn.synsets(form, pos=wn.NOUN)
            if syns:
                return syns[0]
        except WordNetError:
            pass
    return None


# ---------------------------------------------------------------------------
# Core assignment
# ---------------------------------------------------------------------------


def assign_synset(curated_path: str) -> str | None:
    """
    Return the WN synset name string for a single image, or None.

    Follows the six-step priority order described in the module docstring.
    """
    path_lower = curated_path.lower().replace("\\", "/")
    stem, dirname, grandparent = _parse_path(curated_path)
    raw_stem = PureWindowsPath(curated_path).stem.lower()

    # Rule 1: human face — override with gender synset.
    if "human" in path_lower and "face" in path_lower:
        if "female" in raw_stem:
            return "woman.n.01"
        if "male" in raw_stem:
            return "man.n.01"
        # no gender token — fall through to dirname

    # Rule 2: human body man/woman dirs — use dirname identity synset.
    elif "human" in path_lower and "body" in path_lower and dirname in {"man", "woman"}:
        syn = _lookup(dirname)
        return syn.name() if syn is not None else None

    else:
        # Rule 3: manual override.
        if stem in STEM_OVERRIDES:
            return STEM_OVERRIDES[stem]

        # Rule 4: blacklisted stem — skip to dirname.
        if stem not in STEM_BLACKLIST:
            # Rule 5: standard WN stem lookup (space + underscore forms).
            syn = _lookup(stem)
            if syn is not None:
                return syn.name()

    # Rule 6: dirname fallback.
    # If dirname is a colour/generic word, use grandparent instead.
    fallback_dir = grandparent if dirname in STEM_BLACKLIST else dirname
    syn = _lookup(fallback_dir)
    return syn.name() if syn is not None else None


def assign_all(manifest: pd.DataFrame) -> pd.Series:
    """Return a Series of synset names aligned to manifest index."""
    return manifest["curated_path"].map(assign_synset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)

    if "wn_synset_name" in df.columns:
        existing = df["wn_synset_name"].notna().sum()
        print(f"[assign_wn_synsets] Existing wn_synset_name: {existing}/{len(df)} non-null.")

    print(f"[assign_wn_synsets] Assigning synsets for {len(df)} images ...")
    df["wn_synset_name"] = assign_all(df)

    null_count = df["wn_synset_name"].isna().sum()
    if null_count:
        print(f"[assign_wn_synsets] WARNING: {null_count} images have no synset:")
        print(df[df["wn_synset_name"].isna()]["curated_path"].to_string())
    else:
        print("[assign_wn_synsets] All images assigned a synset.")

    n_unique = df["wn_synset_name"].nunique()
    print(f"[assign_wn_synsets] {n_unique} unique synsets across {len(df)} images.")

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"[assign_wn_synsets] Saved -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
