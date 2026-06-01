"""
Shared helpers for building and loading reference RDMs.

All RDMs are stored as condensed upper-triangle vectors (scipy squareform/pdist
convention), length N*(N-1)/2 where N=725. Row/col index i corresponds to row i
of images/manifest.csv (file order). A short hash of the manifest's curated_path
column is stored in each RDM's metadata record and asserted on load, guaranteeing
cross-RDM alignment (and pre/post SHINE correspondence via filename match).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

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
    """
    return pd.read_csv(MANIFEST_PATH)


def order_hash() -> str:
    """
    16-character SHA-256 hex prefix over the manifest's ordered curated_path
    column. Stored with every saved RDM and asserted on load to guarantee
    cross-RDM row/col alignment.
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


def load_image_rgb(path: Path, size: int | None = None) -> np.ndarray:
    """
    Open a PNG, composite any alpha channel over white, and return uint8 RGB.

    Parameters
    ----------
    path : path to image file
    size : if given, assert that the image is size×size (no silent resize)

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
    """
    img = Image.open(path)

    # Composite alpha over white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

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
    except Exception:
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
    state; raises RuntimeError if the manifest has been reordered since the RDM
    was built.

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

    # Check order hash from metadata
    meta_path = RESULTS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            records = json.load(f)
        for r in records:
            if r.get("name") == name:
                stored = r.get("order_hash")
                current = order_hash()
                if stored and stored != current:
                    raise RuntimeError(
                        f"Manifest order hash mismatch for RDM '{name}': "
                        f"stored={stored!r}, current={current!r}. "
                        "The manifest was likely reordered after this RDM was built."
                    )
                break

    return condensed
