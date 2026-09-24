"""
External Kriegeskorte (2008b) / Mur (2013) 92-image data and its overlap with our curated set
(Q1, analysis 2).

The .mat files (rsagroup/rsatoolbox demos/92imageData/, saved under
analysis/data/external/kriegeskorte2008/) carry no image IDs, only 175x175 RGB stimulus arrays and
category flags, so overlap with our Kiani-sourced images can only be found by content. Matching is
average-hash Hamming distance on grayscale, restricted to the same coarse category, assigned
one-to-one (Hungarian) per category. Both sides are composited on the external images' mid-grey
background so backgrounds do not dominate the hash. Nothing here decides which matches are real:
build_overlap_manifest() reports distance, runner-up margin and pixel error per pair for review.

Usage (from repo root):
    from analysis.build_rdms import kriegeskorte92 as k92
    stimuli = k92.load_stimuli()
    matches = k92.build_overlap_manifest(stimuli)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

from analysis.build_rdms.common import REPO_ROOT, image_paths, load_manifest

EXTERNAL_DIR = REPO_ROOT / "analysis" / "data" / "external" / "kriegeskorte2008"
_HASH_SIZE = 16
_GREY = 128
_SIZE = 175

_CATEGORY_TO_CODE = {
    "human face": "HF", "nonhuman face": "AF", "human bodypart": "HB",
    "nonhuman bodypart": "AB", "natural inanimate": "NO", "artificial inanimate": "HO",
}
_KIANI_SOURCE = "Kiani et al., 2007"


@dataclass
class Stimuli:
    images: list[np.ndarray]
    categories: list[str]
    codes: list[str]


def _load_mat(name: str) -> dict:
    return loadmat(EXTERNAL_DIR / name, struct_as_record=False, squeeze_me=True)


def load_stimuli() -> Stimuli:
    """The 92 external stimuli in the .mat file's order, with our 6-way category code for each."""
    objs = _load_mat("Kriegeskorte_Neuron2008_supplementalData.mat")["stimuli_92objs"]
    categories = [str(o.category) for o in objs]
    return Stimuli([np.asarray(o.image) for o in objs], categories, [_CATEGORY_TO_CODE[c] for c in categories])


def load_behavioral_rdms() -> np.ndarray:
    """(16, 92, 92) per-participant similarity-judgment RDMs (Mur et al. 2013)."""
    rdms = _load_mat("92_behavRDMs.mat")["rdms_behav_92"]
    return np.stack([np.asarray(r.RDM, dtype=np.float64) for r in rdms])


def load_brain_rdms() -> dict[str, np.ndarray]:
    """{name: (92, 92)} human IT fMRI RDMs (Kriegeskorte et al. 2008b), one per subject x session."""
    rdms = _load_mat("92_brainRDMs.mat")["RDMs"]
    return {str(r.name): np.asarray(r.RDM, dtype=np.float64) for r in rdms.flat}


def load_our_image(path: Path) -> np.ndarray:
    """Our RGBA image composited on the external mid-grey background, uint8 (H, W, 3)."""
    img = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", img.size, (_GREY, _GREY, _GREY, 255))
    background.alpha_composite(img)
    return np.asarray(background.convert("RGB"))


def _gray_small(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64)


def average_hash(rgb: np.ndarray, size: int = _HASH_SIZE) -> np.ndarray:
    """Boolean (size*size,) hash: grayscale downsample thresholded at its own mean."""
    small = _gray_small(rgb, size)
    return (small > small.mean()).ravel()


def _pixel_mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.BILINEAR))
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def match_by_phash(stimuli: Stimuli, manifest: pd.DataFrame, candidate_rows: pd.Index) -> pd.DataFrame:
    """One-to-one average-hash assignment of external images to candidate manifest rows, per category.

    Returns one row per external image (index into the 92): manifest_row (NaN if its category has
    no candidates), hamming distance out of 256 bits, margin to the best unassigned alternative
    for that image, and pixel MAE of the assigned pair.
    """
    paths = image_paths("pre_shine")
    ours = {row: load_our_image(paths[row]) for row in candidate_rows}
    our_hash = {row: average_hash(img) for row, img in ours.items()}
    records = []
    for code in sorted(set(stimuli.codes)):
        ext_idx = [i for i, c in enumerate(stimuli.codes) if c == code]
        cand = [r for r in candidate_rows if manifest.at[r, "category"] == code]
        if not cand:
            records += [{"ext_index": i, "manifest_row": np.nan} for i in ext_idx]
            continue
        cost = np.array([[np.sum(average_hash(stimuli.images[i]) != our_hash[r]) for r in cand] for i in ext_idx])
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            others = np.delete(cost[a], b)
            records.append({
                "ext_index": ext_idx[a], "manifest_row": cand[b], "hamming": int(cost[a, b]),
                "margin": float(others.min() - cost[a, b]) if others.size else np.nan,
                "pixel_mae": _pixel_mae(stimuli.images[ext_idx[a]], ours[cand[b]]),
            })
        matched = set(rows)
        records += [{"ext_index": ext_idx[a], "manifest_row": np.nan} for a in range(len(ext_idx)) if a not in matched]
    return pd.DataFrame(records).sort_values("ext_index").reset_index(drop=True)


def build_overlap_manifest(stimuli: Stimuli) -> pd.DataFrame:
    """Candidate matches of the 92 external stimuli among our Kiani-2007-sourced images.

    Adds category, external label, curated path and Kiani source filename, plus `flagged`: hamming
    above mean + 2 SD of the assigned distances (the plan's false-positive heuristic). Review the
    distances and pairs yourself: with few true matches the distribution is bimodal and this
    threshold is a coarse guide.
    """
    manifest = load_manifest()
    candidates = manifest.index[manifest["source_dataset"] == _KIANI_SOURCE]
    matches = match_by_phash(stimuli, manifest, candidates)
    matches["ext_category"] = [stimuli.categories[i] for i in matches["ext_index"]]
    assigned = matches["manifest_row"].notna()
    matches["curated_path"] = [
        manifest.at[int(r), "curated_path"] if pd.notna(r) else None for r in matches["manifest_row"]
    ]
    matches["kiani_source_file"] = [
        manifest.at[int(r), "source_filename_or_url"] if pd.notna(r) else None for r in matches["manifest_row"]
    ]
    threshold = matches.loc[assigned, "hamming"].mean() + 2 * matches.loc[assigned, "hamming"].std()
    matches["flagged"] = assigned & (matches["hamming"] > threshold)
    return matches


def match_montage(stimuli: Stimuli, matches: pd.DataFrame, indices: list[int], path: Path) -> None:
    """Side-by-side PNG (external | ours) for the listed rows of `matches`, for visual review."""
    paths = image_paths("pre_shine")
    tiles = []
    for k in indices:
        ext = stimuli.images[int(matches.at[k, "ext_index"])]
        row = matches.at[k, "manifest_row"]
        ours = load_our_image(paths[int(row)]) if pd.notna(row) else np.full_like(ext, 255)
        tiles.append(np.hstack([ext, np.asarray(Image.fromarray(ours).resize((_SIZE, _SIZE)))]))
    Image.fromarray(np.vstack(tiles)).save(path)


# ---------------------------------------------------------------------------
# Approved overlap and external targets, aligned to our manifest order
# ---------------------------------------------------------------------------

_CLEAR_MAX_HAMMING = 10
_CLEAR_MAX_PIXEL_MAE = 5.0


def approved_overlap(matches: pd.DataFrame) -> pd.DataFrame:
    """Matches confirmed by visual review: hamming <= 10 and pixel MAE < 5 (42 of the 92, with a
    clean gap to the next-best assignment), sorted by manifest row.

    Columns ext_index / manifest_row give the aligned orderings used for every subset below.
    """
    clear = matches[(matches["hamming"] <= _CLEAR_MAX_HAMMING) & (matches["pixel_mae"] < _CLEAR_MAX_PIXEL_MAE)]
    out = clear.sort_values("manifest_row").reset_index(drop=True)
    out["manifest_row"] = out["manifest_row"].astype(int)
    return out


def _subset_square(square: np.ndarray, ext_indices: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import squareform

    return squareform(square[np.ix_(ext_indices, ext_indices)], checks=False)


def external_targets(overlap: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    """Condensed external RDMs on the overlap images, ordered like `overlap` (manifest order).

    behavioral_mur2013: 16 participants' similarity-judgment RDMs; hIT_kriegeskorte2008: 8 human IT
    fMRI RDMs (4 subjects x 2 sessions); mIT_kriegeskorte2008: the monkey IT RDM from the paper's
    Figure 1 (a single, pooled RDM).
    """
    from scipy.spatial.distance import squareform

    ext = overlap["ext_index"].to_numpy()
    behavioral = [_subset_square(r, ext) for r in load_behavioral_rdms()]
    brain = [_subset_square(r, ext) for r in load_brain_rdms().values()]
    fig1 = _load_mat("Kriegeskorte_Neuron2008_supplementalData.mat")["RDMs_mIT_hIT_fig1"]
    monkey = [r for r in fig1 if str(r.name).startswith("mIT")]
    monkey_square = [squareform(np.asarray(r.RDM, dtype=np.float64)) for r in monkey]
    return {
        "behavioral_mur2013": behavioral,
        "hIT_kriegeskorte2008": brain,
        "mIT_kriegeskorte2008": [_subset_square(r, ext) for r in monkey_square],
    }
