"""
Shared trial-arrangement rendering, used by both analysis/pilot and
analysis/prod.

Usage (from repo root):
    from analysis.utils.render import render_trial

    img = render_trial(df_subject.iloc[0])   # PIL Image
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

# analysis/utils/render.py → analysis/utils/ → analysis/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

_BG_COLOUR = (250, 250, 250)   # near-white canvas background


def render_trial(
    trial: pd.Series,
    output_width: int = 700,
    output_height: int = 530,
    thumbnail_px: int = 72,
) -> Image.Image:
    """
    Render a single trial's final arrangement as a PIL Image.

    Coordinates in *trial["final_locations"]* are expected to be in [0, 1]
    (screen-independent). The rendered image has size
    (*output_width* × *output_height*) pixels regardless of the subject's
    original screen size.

    Parameters
    ----------
    trial:
        A single row from a trials DataFrame (or any mapping with a
        "final_locations" key holding a JSON string of {"src", "x", "y"}
        items, x/y normalised to [0, 1]).
    output_width, output_height:
        Pixel dimensions of the rendered image.
    thumbnail_px:
        Each stimulus is resized to fit within a *thumbnail_px* × *thumbnail_px*
        bounding box before being pasted onto the canvas.
    """
    locs_raw = trial.get("final_locations", None)
    if pd.isna(locs_raw) or locs_raw == "":
        return Image.new("RGB", (output_width, output_height), _BG_COLOUR)

    locs = json.loads(locs_raw)

    # Add half-a-thumbnail of padding on every side so images placed at the
    # canvas edge aren't clipped when centred on their coordinates.
    pad = thumbnail_px // 2
    canvas = Image.new("RGB", (output_width + 2 * pad, output_height + 2 * pad), _BG_COLOUR)

    for item in locs:
        img_path = _REPO_ROOT / item["src"].lstrip("./")
        try:
            img = Image.open(img_path).convert("RGBA")
        except (FileNotFoundError, OSError):
            continue

        img.thumbnail((thumbnail_px, thumbnail_px), Image.LANCZOS)

        # Map [0, 1] → padded canvas pixel coordinates, centred on the image
        cx = round(item["x"] * output_width) + pad
        cy = round(item["y"] * output_height) + pad
        paste_x = cx - img.width // 2
        paste_y = cy - img.height // 2

        # Composite RGBA (transparent background images) onto canvas
        canvas.paste(img, (paste_x, paste_y), mask=img.split()[3])

    return canvas
