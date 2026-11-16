"""
Pre- vs post-SHINE cohort comparison, external ground truth (analysis/pre_vs_post_shine/
5_ground_truth.ipynb): the Kriegeskorte et al. (2008b) 92-image similarity data (Mur et al. 2013
behavioral ratings; human IT fMRI and monkey IT recordings), restricted to the images it shares with
our curated set.

The .mat files (rsagroup/rsatoolbox, demos/92imageData/, saved under analysis/data/external/
kriegeskorte2008/, gitignored) carry no image IDs, only 175x175 RGB stimulus arrays and category flags,
so the overlap with our Kiani-sourced images was found by content: average-hash Hamming distance on
grayscale, restricted to the same coarse category, assigned one-to-one (Hungarian) per category, both
sides composited on the external images' mid-grey background. propose_matches() is that method. Which
candidates are real matches was decided by visual review (all 42 low-distance pairs approved, the 38
forced leftovers rejected) and is recorded in the tracked gt_overlap_matches.csv, keyed on curated_path;
load_overlap() reads it, so the notebook never re-hashes images. To regenerate: propose_matches() ->
clear_matches() -> review the montage -> write_overlap_csv().

Usage (from repo root):
    from analysis.pre_vs_post_shine import ground_truth as gt
    overlap = gt.load_overlap()
    targets = gt.external_targets(overlap)
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from analysis.build_rdms.behavior.spam_ratings import _combine
from analysis.build_rdms.common import REPO_ROOT, image_paths, load_manifest, subset_condensed
from analysis.pre_vs_post_shine.reference_agreement import (
    TrialsByParticipant, bootstrap_cohort_difference, cohort_difference_equivalence, observed_cohort_fit,
)
from analysis.utils.disk_cache import cached_frame, fingerprint
from analysis.utils.stats import masked_rank_correlation

EXTERNAL_DIR = REPO_ROOT / "analysis" / "data" / "external" / "kriegeskorte2008"
MATCHES_PATH = Path(__file__).parent / "gt_overlap_matches.csv"

_SOURCE_URL = "https://github.com/rsagroup/rsatoolbox/tree/main/demos/92imageData"
_MAT_FILES = (
    "Kriegeskorte_Neuron2008_supplementalData.mat", "92_behavRDMs.mat", "92_brainRDMs.mat", "92_modelRDMs.mat",
)
_HASH_SIZE = 16
_GREY = 128
_SIZE = 175
_CLEAR_MAX_HAMMING = 10
_CLEAR_MAX_PIXEL_MAE = 5.0
_KIANI_SOURCE = "Kiani et al., 2007"
_CATEGORY_TO_CODE = {
    "human face": "HF", "nonhuman face": "AF", "human bodypart": "HB",
    "nonhuman bodypart": "AB", "natural inanimate": "NO", "artificial inanimate": "HO",
}
_MATCH_COLUMNS = ["curated_path", "ext_index", "ext_category", "kiani_source_file", "hamming", "pixel_mae"]


class Stimuli:
    """The 92 external stimuli in the .mat file's order: RGB arrays, category names, our 6-way codes."""

    def __init__(self, images: list[np.ndarray], categories: list[str]) -> None:
        self.images = images
        self.categories = categories
        self.codes = [_CATEGORY_TO_CODE[c] for c in categories]


# ---------------------------------------------------------------------------
# External data
# ---------------------------------------------------------------------------


def _load_mat(name: str) -> dict:
    path = EXTERNAL_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The external 92-image data is not redistributed here: download "
            f"{', '.join(_MAT_FILES)} from {_SOURCE_URL} into {EXTERNAL_DIR}."
        )
    return loadmat(path, struct_as_record=False, squeeze_me=True)


def load_stimuli() -> Stimuli:
    objs = _load_mat("Kriegeskorte_Neuron2008_supplementalData.mat")["stimuli_92objs"]
    return Stimuli([np.asarray(o.image) for o in objs], [str(o.category) for o in objs])


def load_behavioral_rdms() -> np.ndarray:
    """(16, 92, 92) per-participant similarity-judgment RDMs (Mur et al. 2013)."""
    rdms = _load_mat("92_behavRDMs.mat")["rdms_behav_92"]
    return np.stack([np.asarray(r.RDM, dtype=np.float64) for r in rdms])


def load_brain_rdms() -> dict[str, np.ndarray]:
    """{name: (92, 92)} human IT fMRI RDMs (Kriegeskorte et al. 2008b), one per subject x session."""
    rdms = _load_mat("92_brainRDMs.mat")["RDMs"]
    return {str(r.name): np.asarray(r.RDM, dtype=np.float64) for r in rdms.flat}


def load_monkey_rdm() -> np.ndarray:
    """(92, 92) monkey IT RDM (674 neurons) from the paper's Figure 1: a single pooled RDM."""
    fig1 = _load_mat("Kriegeskorte_Neuron2008_supplementalData.mat")["RDMs_mIT_hIT_fig1"]
    (monkey,) = [r for r in fig1 if str(r.name).startswith("mIT")]
    return squareform(np.asarray(monkey.RDM, dtype=np.float64))


# ---------------------------------------------------------------------------
# Matching our images to the external ones
# ---------------------------------------------------------------------------


def load_our_image(path: Path) -> np.ndarray:
    """Our RGBA image composited on the external mid-grey background, uint8 (H, W, 3)."""
    img = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", img.size, (_GREY, _GREY, _GREY, 255))
    background.alpha_composite(img)
    return np.asarray(background.convert("RGB"))


def average_hash(rgb: np.ndarray, size: int = _HASH_SIZE) -> np.ndarray:
    """Boolean (size*size,) hash: grayscale downsample thresholded at its own mean."""
    small = np.asarray(Image.fromarray(rgb).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64)
    return (small > small.mean()).ravel()


def _pixel_mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.BILINEAR))
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def match_by_phash(stimuli: Stimuli, manifest: pd.DataFrame, candidate_rows: pd.Index) -> pd.DataFrame:
    """One-to-one average-hash assignment of external images to candidate manifest rows, per category.

    One row per external image: manifest_row (NaN if its category has no candidates or it lost the
    assignment), hamming distance out of 256 bits, margin to the best alternative, pixel MAE.
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
        records += [{"ext_index": ext_idx[a], "manifest_row": np.nan} for a in range(len(ext_idx)) if a not in set(rows)]
    return pd.DataFrame(records).sort_values("ext_index").reset_index(drop=True)


def propose_matches(stimuli: Stimuli) -> pd.DataFrame:
    """Candidate matches of the 92 external stimuli among our Kiani-2007-sourced images, with the
    curated path, Kiani source file and external category attached. Proposals only: a human decides."""
    manifest = load_manifest()
    matches = match_by_phash(stimuli, manifest, manifest.index[manifest["source_dataset"] == _KIANI_SOURCE])
    matches["ext_category"] = [stimuli.categories[i] for i in matches["ext_index"]]
    for column, source in (("curated_path", "curated_path"), ("kiani_source_file", "source_filename_or_url")):
        matches[column] = [manifest.at[int(r), source] if pd.notna(r) else None for r in matches["manifest_row"]]
    return matches


def clear_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Proposals clearly separated from the forced leftovers (hamming <= 10 and pixel MAE < 5): the
    filter that picked the 42 pairs that were then confirmed by eye."""
    return matches[(matches["hamming"] <= _CLEAR_MAX_HAMMING) & (matches["pixel_mae"] < _CLEAR_MAX_PIXEL_MAE)]


def write_overlap_csv(reviewed: pd.DataFrame, path: Path = MATCHES_PATH) -> None:
    """Record reviewed matches (a propose_matches() subset) as the tracked overlap file."""
    tidy = reviewed.assign(curated_path=reviewed["curated_path"].str.replace("\\", "/", regex=False))[_MATCH_COLUMNS]
    tidy.round({"pixel_mae": 4}).to_csv(path, index=False)


def load_overlap(path: Path = MATCHES_PATH) -> pd.DataFrame:
    """The reviewed overlap, sorted by manifest row, with `manifest_row` derived from `curated_path`.

    Raises if a path is missing from the manifest or duplicated, so a reordered or edited manifest can
    never silently misalign the external RDMs."""
    table = pd.read_csv(path)
    manifest_row = {p.replace("\\", "/"): i for i, p in enumerate(load_manifest()["curated_path"])}
    missing = sorted(set(table["curated_path"]) - set(manifest_row))
    if missing:
        raise ValueError(f"{path.name}: {len(missing)} curated_path value(s) not in the manifest, e.g. {missing[:3]}")
    if table["curated_path"].duplicated().any() or table["ext_index"].duplicated().any():
        raise ValueError(f"{path.name}: duplicated curated_path or ext_index")
    table["manifest_row"] = table["curated_path"].map(manifest_row)
    return table.sort_values("manifest_row").reset_index(drop=True)


# ---------------------------------------------------------------------------
# External targets on the overlap, aligned to our manifest order
# ---------------------------------------------------------------------------


def external_targets(overlap: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    """Condensed external RDMs on the overlap images, ordered like `overlap` (manifest order).

    behavioral_mur2013: 16 raters' similarity judgments; hIT_kriegeskorte2008: 8 human IT fMRI RDMs
    (4 subjects x 2 sessions, so not independent); mIT_kriegeskorte2008: the single pooled monkey RDM.
    """
    ext = overlap["ext_index"].to_numpy()

    def restrict(square: np.ndarray) -> np.ndarray:
        return squareform(square[np.ix_(ext, ext)], checks=False)

    return {
        "behavioral_mur2013": [restrict(r) for r in load_behavioral_rdms()],
        "hIT_kriegeskorte2008": [restrict(r) for r in load_brain_rdms().values()],
        "mIT_kriegeskorte2008": [restrict(load_monkey_rdm())],
    }


def mean_rdm(rdms: list[np.ndarray]) -> np.ndarray:
    """Mean over an external target's RDMs (NaN-ignoring)."""
    return np.nanmean(np.vstack(rdms), axis=0)


def external_leave_one_out(rdms: list[np.ndarray]) -> float:
    """Mean tau-a between each external RDM and the mean of the others: the ceiling for how well a
    single external rater/session is predicted by the group. NaN for a single pooled RDM."""
    if len(rdms) < 2:
        return float("nan")
    return float(np.mean([
        masked_rank_correlation(r, mean_rdm(rdms[:i] + rdms[i + 1:]), tau_a=True)["kendall_a"]
        for i, r in enumerate(rdms)
    ]))


def per_external_fit(perceptual: np.ndarray, rdms: list[np.ndarray]) -> np.ndarray:
    """tau-a of one cohort's combined RDM against each external RDM separately."""
    return np.array([masked_rank_correlation(perceptual, r, tau_a=True)["kendall_a"] for r in rdms])


def calibrate_equivalence_bound(
    trials: TrialsByParticipant, references: dict[str, np.ndarray], n_conds: int, rows: np.ndarray,
    *, n_splits: int = 10, n_boot: int = 200, seed: int = 100, cache_path: Path | None = None,
) -> pd.DataFrame:
    """The equivalence procedure applied to random halves of ONE cohort, where the true difference is
    zero by construction: how often does the project's bound rule declare equivalence (target 80%)?

    Halves have about half the participants, so their SE and bound are larger than in the real
    comparison; this checks the rule, not the real bound. Returns the equivalence table with a `split` column."""
    def run() -> pd.DataFrame:
        ids = list(trials)
        rng = np.random.default_rng(seed)
        out = []
        for split in range(n_splits):
            order = rng.permutation(len(ids))
            half_a = {ids[i]: trials[ids[i]] for i in order[: len(ids) // 2]}
            half_b = {ids[i]: trials[ids[i]] for i in order[len(ids) // 2:]}
            rdm_a, rdm_b = (
                subset_condensed(_combine([t for v in half.values() for t in v], n_conds, "simple")[0], rows)
                for half in (half_a, half_b)
            )
            boots = bootstrap_cohort_difference(
                half_a, half_b, references, n_conds, n_boot=n_boot, seed=seed + split + 1, subset=rows,
            )
            table = cohort_difference_equivalence(
                {"own": observed_cohort_fit(rdm_a, rdm_b, references)}, boots, len(half_a), len(half_b),
            )
            out.append(table.assign(split=split))
        return pd.concat(out, ignore_index=True)

    key = fingerprint(list(trials), list(references), *references.values(), n_conds, rows, n_splits, n_boot, seed)
    return cached_frame(cache_path, key, run)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_match_grid(stimuli: Stimuli, overlap: pd.DataFrame, *, per_row: int = 6, size: int = 88) -> go.Figure:
    """Review montage of the overlap: each tile is the external image next to our (pre-SHINE) image."""
    paths = image_paths("pre_shine")
    tiles = []
    for _, row in overlap.iterrows():
        ext = np.asarray(Image.fromarray(stimuli.images[int(row["ext_index"])]).resize((size, size)))
        ours = np.asarray(Image.fromarray(load_our_image(paths[int(row["manifest_row"])])).resize((size, size)))
        tiles.append(np.hstack([ext, ours]))
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])
    fig = go.Figure(go.Image(z=grid))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=grid.shape[0] * 1.3, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                      title="Each tile: 92-image dataset (left) | ours, pre-SHINE (right)")
    return fig
