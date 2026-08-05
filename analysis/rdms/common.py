"""
Shared helpers for building and loading reference RDMs.

All RDMs are stored as condensed upper-triangle vectors (scipy squareform/pdist
convention), length N*(N-1)/2 where N=725. Row/col index i corresponds to row i
of images/manifest.csv (file order). A short hash of the manifest's curated_path
column is stored in each RDM's metadata record and asserted on load, guaranteeing
cross-RDM alignment (and pre/post SHINE correspondence via filename match).

Embedder modules (e.g. clip.py) additionally persist the raw (N, embedding_dim)
embedding matrix behind the RDM, via save_embeddings()/load_embeddings(), as
E_<name>.npy + embeddings_metadata.json. Embeddings are the non-lossy artifact —
any pairwise-distance metric can be re-derived from them without re-running the
(expensive) encoder forward pass. euclidean_distances()/cosine_distances() are
shared, embedder-agnostic helpers for that derivation step, with an optional
save_result=True to persist the resulting RDM via save_rdm() in the same call.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# analysis/rdms/common.py  ->  analysis/rdms/  ->  analysis/  ->  repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "images" / "manifest.csv"
IMAGES_ROOT = REPO_ROOT / "images"
RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "rdms"

_EXPECTED_N = 725
_EXPECTED_LEN = _EXPECTED_N * (_EXPECTED_N - 1) // 2  # 262_450

# ---------------------------------------------------------------------------
# Manifest / image order
# ---------------------------------------------------------------------------


def load_manifest() -> pd.DataFrame:
    """
    Load images/manifest.csv in file row order.

    This row order is the canonical image index: row i <-> index i in every
    condensed RDM vector.

    Raises ValueError if curated_path has duplicates. A duplicated path is
    silently destructive: every builder still sees _EXPECTED_N rows and every
    path still resolves to a real file, so one image is encoded twice and
    another is dropped without any downstream check noticing. Set-equality
    against the image directories is the stronger check but requires disk
    access; see check_manifest_matches_disk() in validate_rdms.py.
    """
    df = pd.read_csv(MANIFEST_PATH)
    dupes = df["curated_path"][df["curated_path"].duplicated(keep=False)]
    if not dupes.empty:
        listed = "\n  ".join(sorted(dupes.unique()))
        raise ValueError(
            f"{MANIFEST_PATH} has {dupes.nunique()} duplicated curated_path "
            f"value(s), so it indexes fewer than {len(df)} distinct images:\n"
            f"  {listed}"
        )
    return df


@lru_cache(maxsize=1)
def order_hash() -> str:
    """
    16-character SHA-256 hex prefix over the manifest's ordered curated_path
    column. Stored with every saved RDM and asserted on load to guarantee
    cross-RDM row/col alignment.

    Result is cached; the manifest row order must not change between a
    save_rdm() call and the subsequent load_rdm() call.
    """
    df = load_manifest()
    paths_str = "\n".join(df["curated_path"].tolist())
    return hashlib.sha256(paths_str.encode()).hexdigest()[:16]


def image_paths(variant: str) -> list[Path]:
    """
    Return the list of absolute image Paths in manifest order for *variant*.

    Parameters
    ----------
    variant : 'pre_shine' or 'post_shine'
    """
    if variant not in ("pre_shine", "post_shine"):
        raise ValueError(f"variant must be 'pre_shine' or 'post_shine', got {variant!r}")
    df = load_manifest()
    root = IMAGES_ROOT / variant
    # curated_path uses Windows backslashes; normalise to forward slashes for pathlib
    return [root / p.replace("\\", "/") for p in df["curated_path"]]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def open_as_rgb_pil(path: Path) -> Image.Image:
    """
    Open an image file, composite any alpha channel over white, and return a
    PIL RGB image.

    Plain RGB images are returned directly (no conversion overhead).
    All other modes (RGBA, L, LA, P, PA, …) are converted to RGBA first,
    which correctly applies palette lookups and transparency, then
    composited over a white background.
    """
    img = Image.open(path)
    if img.mode == "RGB":
        return img
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg.convert("RGB")


def load_image_rgb(path: Path, size: int | None = None) -> np.ndarray:
    """
    Open an image, composite alpha over white, and return a uint8 RGB array.

    Parameters
    ----------
    path : path to image file
    size : if given, assert that the image is size×size (no silent resize)

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
    """
    img = open_as_rgb_pil(path)
    if size is not None and img.size != (size, size):
        raise ValueError(
            f"Expected {size}x{size} image, got {img.size[0]}x{img.size[1]} for {path}"
        )
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# RDM save / load
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return "unknown"


def save_rdm(
    name: str,
    condensed: np.ndarray,
    *,
    metric: str,
    source: str,
    extra: dict | None = None,
) -> Path:
    """
    Save a condensed RDM vector as analysis/results/rdms/D_<name>.npy and record
    provenance in metadata.json in the same directory.

    Parameters
    ----------
    name      : file stem suffix; output is D_<name>.npy
    condensed : 1-D condensed upper-triangle vector, length 262_450
    metric    : human-readable metric (e.g. 'euclidean', 'cosine')
    source    : module that produced this RDM
    extra     : additional key-value metadata to include in the record

    Returns
    -------
    Path of the written .npy file
    """
    if condensed.ndim != 1 or len(condensed) != _EXPECTED_LEN:
        raise ValueError(
            f"Expected 1-D condensed vector of length {_EXPECTED_LEN}, "
            f"got shape {condensed.shape}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"D_{name}.npy"
    np.save(out_path, condensed.astype(np.float64))

    # Update metadata.json (remove stale record for same name, then append)
    meta_path = RESULTS_DIR / "metadata.json"
    records: list[dict] = []
    if meta_path.exists():
        with open(meta_path) as f:
            records = json.load(f)
    records = [r for r in records if r.get("name") != name]
    records.append({
        "name": name,
        "file": f"D_{name}.npy",
        "metric": metric,
        "source": source,
        "n_images": _EXPECTED_N,
        "condensed_length": _EXPECTED_LEN,
        "order_hash": order_hash(),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    })
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)

    return out_path


def load_rdm(name: str) -> np.ndarray:
    """
    Load condensed RDM vector from D_<name>.npy.

    Asserts that the condensed length and manifest order_hash match the current
    state. Raises RuntimeError if metadata.json is absent or has no record for
    this name (both indicate the file was not written via save_rdm()).

    Returns
    -------
    np.ndarray, 1-D float64 condensed vector of length 262_450
    """
    path = RESULTS_DIR / f"D_{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"RDM not found: {path}")

    condensed = np.load(path)
    if len(condensed) != _EXPECTED_LEN:
        raise ValueError(
            f"Loaded RDM '{name}' has length {len(condensed)}, expected {_EXPECTED_LEN}"
        )

    # Enforce hash guard — both missing-metadata and missing-record are errors
    meta_path = RESULTS_DIR / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"metadata.json not found in {RESULTS_DIR}. "
            "Load only RDMs that were written via save_rdm()."
        )
    with open(meta_path) as f:
        records = json.load(f)
    for r in records:
        if r.get("name") == name:
            stored = r.get("order_hash")
            current = order_hash()
            if stored is not None and stored != current:
                raise RuntimeError(
                    f"Manifest order hash mismatch for RDM '{name}': "
                    f"stored={stored!r}, current={current!r}. "
                    "The manifest was likely reordered after this RDM was built."
                )
            break
    else:
        raise RuntimeError(
            f"No metadata record found for RDM '{name}' in {meta_path}. "
            "Load only RDMs that were written via save_rdm()."
        )

    return condensed


# ---------------------------------------------------------------------------
# Embedding save / load
# ---------------------------------------------------------------------------


def save_embeddings(
    name: str,
    embeddings: np.ndarray,
    *,
    source: str,
    extra: dict | None = None,
) -> Path:
    """
    Save a raw embedding matrix as analysis/results/rdms/E_<name>.npy and record
    provenance in embeddings_metadata.json in the same directory.

    Parameters
    ----------
    name       : file stem suffix; output is E_<name>.npy
    embeddings : 2-D array, shape (725, embedding_dim)
    source     : module that produced these embeddings
    extra      : additional key-value metadata to include in the record

    Returns
    -------
    Path of the written .npy file
    """
    if embeddings.ndim != 2 or embeddings.shape[0] != _EXPECTED_N:
        raise ValueError(
            f"Expected 2-D embedding matrix with {_EXPECTED_N} rows, "
            f"got shape {embeddings.shape}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"E_{name}.npy"
    np.save(out_path, embeddings.astype(np.float32))

    # Update embeddings_metadata.json (remove stale record for same name, then append)
    meta_path = RESULTS_DIR / "embeddings_metadata.json"
    records: list[dict] = []
    if meta_path.exists():
        with open(meta_path) as f:
            records = json.load(f)
    records = [r for r in records if r.get("name") != name]
    records.append({
        "name": name,
        "file": f"E_{name}.npy",
        "source": source,
        "n_images": _EXPECTED_N,
        "embedding_dim": embeddings.shape[1],
        "order_hash": order_hash(),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    })
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)

    return out_path


def load_embeddings(name: str) -> np.ndarray:
    """
    Load embedding matrix from E_<name>.npy.

    Asserts that the row count and manifest order_hash match the current
    state. Raises RuntimeError if embeddings_metadata.json is absent or has
    no record for this name (both indicate the file was not written via
    save_embeddings()).

    Returns
    -------
    np.ndarray, 2-D float32 matrix, shape (725, embedding_dim)
    """
    path = RESULTS_DIR / f"E_{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}")

    embeddings = np.load(path)
    if embeddings.ndim != 2 or embeddings.shape[0] != _EXPECTED_N:
        raise ValueError(
            f"Loaded embeddings '{name}' have shape {embeddings.shape}, "
            f"expected ({_EXPECTED_N}, *)"
        )

    # Enforce hash guard — both missing-metadata and missing-record are errors
    meta_path = RESULTS_DIR / "embeddings_metadata.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"embeddings_metadata.json not found in {RESULTS_DIR}. "
            "Load only embeddings that were written via save_embeddings()."
        )
    with open(meta_path) as f:
        records = json.load(f)
    for r in records:
        if r.get("name") == name:
            stored = r.get("order_hash")
            current = order_hash()
            if stored is not None and stored != current:
                raise RuntimeError(
                    f"Manifest order hash mismatch for embeddings '{name}': "
                    f"stored={stored!r}, current={current!r}. "
                    "The manifest was likely reordered after these embeddings were built."
                )
            break
    else:
        raise RuntimeError(
            f"No metadata record found for embeddings '{name}' in {meta_path}. "
            "Load only embeddings that were written via save_embeddings()."
        )

    return embeddings


# ---------------------------------------------------------------------------
# Shared pairwise-distance helpers (cross-embedder)
# ---------------------------------------------------------------------------


def _pairwise_distances(
    embeddings: np.ndarray,
    *,
    metric: str,
    save_result: bool,
    name: str | None,
    source: str | None,
    extra: dict | None,
) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D embedding matrix, got shape {embeddings.shape}")
    condensed = pdist(embeddings, metric=metric)
    if save_result:
        if name is None or source is None:
            raise ValueError("name and source are required when save_result=True")
        save_rdm(name, condensed, metric=metric, source=source, extra=extra)
    return condensed


def euclidean_distances(
    embeddings: np.ndarray,
    *,
    save_result: bool = False,
    name: str | None = None,
    source: str | None = None,
    extra: dict | None = None,
) -> np.ndarray:
    """
    Condensed pairwise Euclidean distance vector over an (N, D) embedding matrix.

    Parameters
    ----------
    embeddings  : 2-D array, shape (725, D) — any embedder's feature matrix
    save_result : if True, persist the result via save_rdm() as D_<name>.npy
                  (requires `name` and `source`)
    name, source, extra : forwarded to save_rdm() when save_result=True

    Returns
    -------
    np.ndarray, 1-D condensed vector, length 262_450
    """
    return _pairwise_distances(
        embeddings, metric="euclidean",
        save_result=save_result, name=name, source=source, extra=extra,
    )


def cosine_distances(
    embeddings: np.ndarray,
    *,
    save_result: bool = False,
    name: str | None = None,
    source: str | None = None,
    extra: dict | None = None,
) -> np.ndarray:
    """
    Condensed pairwise cosine distance vector (1 - cosine similarity) over an
    (N, D) embedding matrix.

    Parameters
    ----------
    embeddings  : 2-D array, shape (725, D) — any embedder's feature matrix
    save_result : if True, persist the result via save_rdm() as D_<name>.npy
                  (requires `name` and `source`)
    name, source, extra : forwarded to save_rdm() when save_result=True

    Returns
    -------
    np.ndarray, 1-D condensed vector, length 262_450
    """
    return _pairwise_distances(
        embeddings, metric="cosine",
        save_result=save_result, name=name, source=source, extra=extra,
    )
