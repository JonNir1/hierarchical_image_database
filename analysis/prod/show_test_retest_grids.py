"""
Render final-arrangement grids for the test-retest (repeated-trial) subset of
data/prod sessions.

For every completed prod session (20 experimental trials + 4 catch trials),
each of the 3 verbatim-repeat pairs (trial_6<-trial_1|2|3, trial_11<-...,
trial_16<-...) is rendered as a two-column panel: original trial (left) vs.
its repeat (right), annotated with the pair's Spearman R (computed over the
190 pairwise image distances). One PNG per subject is written to
analysis/prod/figures/, titled with the subject's mean R across
their 3 pairs.

Run from repo root:
    .venv/Scripts/python analysis/prod/show_test_retest_grids.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

from scipy.stats import spearmanr
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from analysis.utils.render import render_trial  # noqa: E402

BASE = _REPO_ROOT / "data" / "prod"
OUT_DIR = Path(__file__).resolve().parent / "figures"

PANEL_W, PANEL_H, THUMB_PX = 480, 360, 55
SUBPLOT_TITLE_H = 34
R_COLUMN_W = 170
ROW_GAP = 18
HEADER_H = 64
MARGIN = 16


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_complete(rows: list[dict]) -> bool:
    trial_nums, catch_nums = set(), set()
    for r in rows:
        tt = r.get("trial_type", "")
        m = re.fullmatch(r"trial_(\d+)", tt)
        if m:
            trial_nums.add(int(m.group(1)))
        m2 = re.fullmatch(r"catch_(\d+)", tt)
        if m2:
            catch_nums.add(int(m2.group(1)))
    return trial_nums == set(range(1, 21)) and catch_nums == set(range(1, 5))


def parse_trials(rows: list[dict]) -> dict[int, dict]:
    trials = {}
    for r in rows:
        m = re.fullmatch(r"trial_(\d+)", r.get("trial_type", ""))
        if m:
            trials[int(m.group(1))] = r
    return trials


def image_set(row: dict) -> frozenset:
    return frozenset(loc["src"] for loc in json.loads(row["init_locations"]))


def distance_dict(row: dict) -> dict[frozenset, float]:
    pd = json.loads(row["pairwise_distances"])
    return {frozenset((d["src1"], d["src2"])): d["distance"] for d in pd}


def normalised_row(row: dict) -> dict:
    """Copy of *row* with final_locations pixel coords mapped to [0, 1]
    using this trial's own sort_area_width/height (render_trial expects
    coordinates already normalised, matching analysis/pilot/parser.py)."""
    w, h = float(row["sort_area_width"]), float(row["sort_area_height"])
    locs = json.loads(row["final_locations"])
    for item in locs:
        item["x"] = item["x"] / w
        item["y"] = item["y"] / h
    out = dict(row)
    out["final_locations"] = json.dumps(locs)
    return out


def build_subject_figure(pid: str, pairs: list[dict]) -> Image.Image:
    """pairs: list of {"orig_num", "repeat_num", "orig_row", "repeat_row", "r"}."""
    n_rows = len(pairs)
    width = MARGIN * 2 + PANEL_W * 2 + R_COLUMN_W
    height = (
        HEADER_H
        + n_rows * (SUBPLOT_TITLE_H + PANEL_H)
        + (n_rows - 1) * ROW_GAP
        + MARGIN
    )

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
        subtitle_font = ImageFont.truetype("arial.ttf", 16)
        r_font = ImageFont.truetype("arialbd.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default(size=22)
        subtitle_font = ImageFont.load_default(size=16)
        r_font = ImageFont.load_default(size=18)

    mean_r = statistics.mean(p["r"] for p in pairs)
    title = f"{pid}  —  overall Spearman R = {mean_r:.3f}"
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, 18), title, font=title_font, fill=(20, 20, 20))

    y = HEADER_H
    for pair in pairs:
        left_img = render_trial(
            normalised_row(pair["orig_row"]),
            output_width=PANEL_W, output_height=PANEL_H, thumbnail_px=THUMB_PX,
        )
        right_img = render_trial(
            normalised_row(pair["repeat_row"]),
            output_width=PANEL_W, output_height=PANEL_H, thumbnail_px=THUMB_PX,
        )

        left_title = f"trial {pair['orig_num']}"
        right_title = f"trial {pair['repeat_num']}"
        lw = draw.textlength(left_title, font=subtitle_font)
        rw = draw.textlength(right_title, font=subtitle_font)
        draw.text(
            (MARGIN + (PANEL_W - lw) / 2, y + (SUBPLOT_TITLE_H - 16) / 2),
            left_title, font=subtitle_font, fill=(60, 60, 60),
        )
        draw.text(
            (MARGIN + PANEL_W + (PANEL_W - rw) / 2, y + (SUBPLOT_TITLE_H - 16) / 2),
            right_title, font=subtitle_font, fill=(60, 60, 60),
        )

        img_y = y + SUBPLOT_TITLE_H
        canvas.paste(left_img, (MARGIN, img_y))
        canvas.paste(right_img, (MARGIN + PANEL_W, img_y))

        r_text = f"R = {pair['r']:.3f}"
        rtw = draw.textlength(r_text, font=r_font)
        r_x = MARGIN + PANEL_W * 2 + (R_COLUMN_W - rtw) / 2
        r_y = img_y + PANEL_H / 2 - 10
        draw.text((r_x, r_y), r_text, font=r_font, fill=(0, 90, 160))

        y += SUBPLOT_TITLE_H + PANEL_H + ROW_GAP

    return canvas


def main() -> None:
    files = sorted(glob.glob(str(BASE / "*.csv")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for f in files:
        rows = load_rows(f)
        if not rows or not is_complete(rows):
            continue
        pid = rows[0]["participant_id"].strip()
        trials = parse_trials(rows)

        pairs = []
        for num in sorted(trials):
            row = trials[num]
            if row.get("is_trial_repeat", "").strip().lower() != "true":
                continue
            orig_num = int(float(row["repeat_of_trial_number"]))
            orig_row = trials[orig_num]

            assert image_set(row) == image_set(orig_row), (
                f"{pid}: trial {num} does not reproduce trial {orig_num}'s image set"
            )

            d_orig, d_repeat = distance_dict(orig_row), distance_dict(row)
            keys = sorted(d_orig.keys() & d_repeat.keys(), key=lambda k: sorted(k))
            r, _p = spearmanr([d_orig[k] for k in keys], [d_repeat[k] for k in keys])

            pairs.append({
                "orig_num": orig_num, "repeat_num": num,
                "orig_row": orig_row, "repeat_row": row, "r": r,
            })

        if not pairs:
            continue

        fig = build_subject_figure(pid, pairs)
        out_path = OUT_DIR / f"{pid}.png"
        fig.save(out_path)
        written.append(out_path)
        print(f"Wrote {out_path}  ({len(pairs)} pairs, mean R={statistics.mean(p['r'] for p in pairs):.3f})")

    print(f"\n{len(written)} subject figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
