# SpAM Task — Quickstart Guide

This guide walks you through adapting and deploying the SpAM (Spatial Arrangement Method)
task for your own image dataset. No JavaScript experience is required; most of the work is
configuration and file organisation.

---

## What the task does

Participants see a canvas containing several object images and drag them into spatial positions
that reflect perceived visual similarity — similar images close together, dissimilar images far
apart. Pairwise Euclidean distances between final positions are recorded as similarity scores.

Each participant sees a random subset of your image library across multiple trials. The full
dataset is covered by pooling data across participants. The output is a pairwise dissimilarity
matrix ready for Multidimensional Scaling (MDS) or other analyses.

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python 3.10+ | Run `generate_manifest.py` |
| A local HTTP server | Test the task in a browser (`python -m http.server`) |
| A Pavlovia account | Host the experiment online |
| A Prolific account | Recruit participants |

No Python packages beyond the standard library are needed for `generate_manifest.py`.

---

## Step 1 — Prepare your images

The task expects two sets of PNG images. Create a directory for each (they can live anywhere
on your machine; you will point `task_config.json` at them in Step 2).

### Main stimuli
Your experimental image library — the images you want similarity data for. There is no hard
limit on how many images you can have; the task samples a subset per participant so that the
full library is covered across participants. Images should:
- Be PNG format with a transparent or uniform background
- Be similar in size (the task renders them all at the same display size, so wildly different
  aspect ratios will look odd)

### Practice + catch images
One shared directory, used for two purposes:
- **Practice**: a small unrecorded warm-up trial (~8 images) using familiar, everyday objects —
  participants use it to learn the drag interface before real data collection begins. **These
  images are not part of your dataset** and the trial is discarded.
- **Catch**: attention checks (~20 or more images) used to check whether participants are
  following instructions. On catch trials, participants are told to place all images in a
  specific region of the screen (e.g. "top left corner"). Responses that ignore the instruction
  are flagged.

Use images that are distinctive enough that participants have no reason to be confused — simple
emoji-style or clip-art images work well.

---

## Step 2 — Configure the experiment

Open `SpAM_Task/task_config.json` in a text editor. The parameters you are most likely to need to
change are:

```json
{
  "deployment": {
    "debug_shine_variant": "pre"
  },
  "experimental_trials": {
    "stimuli_path": "./images/"
  },
  "catch_trials": {
    "stimuli_path": "SpAM_Task/assets/openmoji/"
  },
  "screening_block": {
    "prolific_code": ""
  },
  "experimental_block": {
    "prolific_code": ""
  },
  "consent": {
    "no_consent_code": ""
  }
}
```

**Dataset layout.** Main stimuli live at `<repo>/images/{pre_shine,post_shine}/` (one
directory per SHINE-preprocessing variant). The main directory is resolved as
`<stimuli_path>/<shine_variant>_shine/`. `experimental_trials.stimuli_path` is relative to the
**project root** (matching the browser's resolution against `<repo>/index.html`), so the default
`"./images/"` reaches `<repo>/images/`.

The dataset (725 images per variant) is tracked on both `main` (GitHub) and
`pavlovia_deploy` (Pavlovia gitlab) — no force-add required. Source datasets used
to derive the curated set live under `<repo>/source_datasets/` but are not
redistributed.

To populate the dataset, copy or move your images into `<repo>/images/pre_shine/`
(and later `<repo>/images/post_shine/`), preserving any category subdirectories. No
junctions or symlinks are required.

**SHINE variant.** In `"debug"` mode, set `deployment.debug_shine_variant` to `"pre"` to load
original images or `"post"` to load SHINE-preprocessed images. In `"pilot"` mode the variant is
always `"pre"`; in `"production"` mode it's assigned per-participant from a hash of their
Prolific PID. The selected variant is recorded in the manifest and in every saved trial row.

**Leave `screening_block.prolific_code`, `experimental_block.prolific_code`, and
`consent.no_consent_code` empty for now** — you will fill them in after creating your Prolific
study (Step 6). These are the three Prolific completion codes for the task's exit paths: screened
out, completed, and declined consent respectively — see "Two-stage session structure" in
`CLAUDE.md`.

### Tuning the trial design

The defaults below work for a dataset of ~750 images with 50–75 participants. If your dataset
is substantially smaller or larger, adjust accordingly. `screening_block` and `experimental_block`
share the same shape (`screening_block` additionally has `enabled`/`thresholds`); a stage's total
main trials = `num_experimental_trials + num_repeat_trials`, plus `num_catch_trials` catch trials.

| Parameter | Default | What it controls |
|---|---|---|
| `screening_block.enabled` | `true` | Whether the optional screening stage runs at all before the experimental stage. If `false`, every participant goes straight to `experimental_block`. |
| `screening_block` / `experimental_block` . `num_experimental_trials` | 6 / 12 | How many *distinct* sorting trials that stage presents |
| `experimental_trials.images_per_trial` | 20 | How many images appear on screen per trial (shared by both stages) |
| `screening_block` / `experimental_block` . `num_repeat_trials` | 2 / 2 | Additional (additive) verbatim repeats of that stage's own distinct trials — same k images, reshuffled order — for test-retest reliability of the arrangement itself. The only way an image can appear more than once per subject, and always scoped to its own stage. Must be in `[0, num_experimental_trials]`. |
| `screening_block` / `experimental_block` . `min_repeat_separation` | 2 / 3 | Minimum number of other stage-local main-trial slots between an original trial and its verbatim repeat (prevents back-to-back identical trials). May be 0 (no constraint). |
| `experimental_trials.min_trial_duration_ms` | 60000 | UI-enforced minimum: "Done" button is disabled for this many ms on main trials (live countdown shown). No post-hoc RT flag — UI enforcement makes it unnecessary. |
| `display.min_header_height_px` | 80 | Minimum viewport height (px) reserved above the sort area for the prompt text. Increase if your prompt text wraps to more lines than the default budget expects. |
| `display.min_footer_height_px` | 80 | Minimum viewport height (px) reserved below the sort area for the counter + Done button (and the catch-trial compliance warning line). Increase if this content is clipped or forces a scrollbar. |
| `screening_block` / `experimental_block` . `num_catch_trials` | 1 / 2 | How many attention-check trials are interleaved in that stage |
| `catch_trials.images_per_trial` | 10 | Images per catch trial (shared by both stages) |

> **Coverage check**: with your chosen parameters, verify that each image will be seen by
> enough participants for reliable averaging.
> Expected views per image ≈ `N_subjects × ((screening.num_experimental_trials +
> experimental.num_experimental_trials) × images_per_trial) / total_images` — only *distinct*
> trials draw new images, so repeats don't add coverage.
> Aim for at least 10–15 views per image.

### Quality control thresholds

| Parameter | Default | Meaning |
|---|---|---|
| `experimental_trials.min_pairwise_distance_sd` | 0.1 | Main trials where all images are piled in one spot are flagged |
| `experimental_trials.min_move_item_ratio` | 0.65 | Minimum ratio of drag-end events to images shown (`num_moves / numItems`) for a main trial to pass; flags main trials where the participant barely engaged |
| `catch_trials.location_tolerance` | 0.15 | Catch trial: every individual image must be within this normalised distance of the target region. Live-enforced (Done button stays disabled until satisfied), not a post-hoc QC flag. |

Catch trials have no post-hoc `qc_flag` — see "Quality control — catch trials" in `CLAUDE.md`.

These two per-trial flags are diagnostic only. Actual exclusion happens once, at the end of the
screening stage, via `screening_block.thresholds` (fail-rate and test-retest-reliability
thresholds evaluated across that stage's own trials — see "Live screening" in `CLAUDE.md`). You
can leave everything at its defaults for a pilot run and revisit after inspecting the data.

---

## Step 3 — Generate the stimulus manifest

The browser cannot list the contents of a server directory, so the task needs a pre-built
index of your images. Generate it by running:

```bash
cd SpAM_Task
python generate_manifest.py
```

This reads your paths from `task_config.json`, scans each directory recursively for `.png` files,
and writes `stimuli_manifest.json`. Run it again any time you add, remove, or rename images.

You should see output like:
```
images (pre-SHINE): 725 images in <repo>/images/pre_shine
practice_images: 8 images in <repo>/SpAM_Task/assets/openmoji
catch_images: 22 images in <repo>/SpAM_Task/assets/openmoji

Manifest written to .../SpAM_Task/stimuli_manifest.json
```

The script aborts if `stimuli_path/<variant>_shine/` is missing or empty (main set is
fatal). Practice/catch paths emit warnings on missing content but do not abort.

---

## Step 4 — Install jsPsych locally

The task loads jsPsych from a local `jspsych/` directory (no CDN dependency). Download the
following files and place them in `SpAM_Task/jspsych/`:

| File | Source |
|---|---|
| `jspsych.js` | [github.com/jspsych/jsPsych](https://github.com/jspsych/jsPsych/releases) — jsPsych 7 |
| `plugin-free-sort-patched.js` | Based on jsPsych 7 release — locally patched to add move timestamps (see file header) |
| `plugin-preload.js` | Same release |
| `plugin-html-button-response.js` | Same release |
| `plugin-fullscreen.js` | Same release |
| `seedrandom.js` | [github.com/davidbau/seedrandom](https://github.com/davidbau/seedrandom) — v3 |
| `jspsych-7-pavlovia-2022.1.1.js` | [gitlab.pavlovia.org/tpronk/jsPsych-7-pavlovia-2021.12](https://gitlab.pavlovia.org/tpronk/jsPsych-7-pavlovia-2021.12) |

> **Version note**: the task was built against jsPsych 7. Do not use jsPsych 6 — the plugin
> API is incompatible.

---

## Step 5 — Test locally

1. Enable debug mode in `task_config.json` (bypasses the Prolific ID requirement):
   ```json
   "mode": "debug"
   ```

2. Start a local web server from the **repo root** (where `index.html` lives):
   ```bash
   python -m http.server 8000
   ```

3. Open your browser and go to:
   ```
   http://localhost:8000
   ```
   In debug mode the task uses a fixed participant ID (`debug_participant`) and logs
   trial-level QC information to the browser console (F12 → Console tab).

4. Complete the task end-to-end. At the end a CSV file will be downloaded automatically —
   this is your local stand-in for the Pavlovia data save.

5. When you are satisfied, set `deployment.mode` to `"pilot"` (first real-user run)
   or `"production"` (full data collection) before deploying.

---

## Step 6 — Deploy to Pavlovia

1. **Create a Pavlovia project** at [pavlovia.org](https://pavlovia.org) and add it as a
   git remote named `pavlovia`. Then push the `pavlovia_deploy` branch to it using the
   deploy script (from anywhere inside the repo, on `main`):
   ```bash
   # macOS/Linux/Git Bash
   bash SpAM_Task/scripts/deploy_pavlovia.sh
   ```
   ```powershell
   # Windows PowerShell
   .\SpAM_Task\scripts\deploy_pavlovia.ps1
   ```
   The script rebuilds `pavlovia_deploy` from `main` and pushes it to Pavlovia. The branch
   ships only `index.html`, `SpAM_Task/`, and `images/` — all other directories are excluded
   automatically via an include-based `.gitignore` block appended by the script. Run the same
   script any time you want to sync Pavlovia with `main`.

2. **Activate the experiment**: switch the project status to *Piloting* for testing or
   *Running* for live data collection.

4. **Copy the Pavlovia experiment URL** — it will look like:
   ```
   https://run.pavlovia.org/<your-username>/<project-name>/?PROLIFIC_PID={{%PROLIFIC_PID%}}
   ```
   The `{{%PROLIFIC_PID%}}` part is a Prolific template variable — leave it exactly as shown.

---

## Step 7 — Set up Prolific recruitment

The task has three distinct exit paths, each with its own Prolific completion code: participants
who complete the full task, participants screened out at the end of the screening stage (only
possible if `screening_block.enabled` is `true`), and participants who decline consent. You need
a separate Prolific study *completion code* for each — see "Live screening" in `CLAUDE.md` for how
the screen-out path is triggered.

1. Log in to [prolific.com](https://www.prolific.com) and create a new study.
2. Paste the Pavlovia URL (from Step 6) into the *Study URL* field.
3. Under *Completion*, select *I'll redirect them using a URL* and set up your codes/redirects for
   the three outcomes above. Each yields a completion URL of the form
   `https://app.prolific.com/submissions/complete?cc=XXXXXXXX` — you only need the trailing
   `cc=` code, not the full URL.
4. Paste the three codes into `task_config.json`:
   ```json
   "screening_block":   { "prolific_code": "XXXXXXXX" },
   "experimental_block": { "prolific_code": "XXXXXXXX" },
   "consent":           { "no_consent_code": "XXXXXXXX" }
   ```
5. Commit the updated `task_config.json` to `main`, push to GitHub, then run
   `bash SpAM_Task/scripts/deploy_pavlovia.sh` (or `.\SpAM_Task\scripts\deploy_pavlovia.ps1` on
   Windows PowerShell) to sync Pavlovia. Re-activate the experiment afterwards.

**Recommended pilot**: run 10–15 participants before launching full data collection. Inspect
the downloaded CSVs for QC flag rates and check that the MDS output shows sensible structure.

---

## Step 8 — Collect and download data

Data files are saved automatically by Pavlovia at the end of each session. Download them from
your Pavlovia project dashboard under *Data → Results*. Each file is a CSV named by session
ID. Pass the folder of CSVs to the post-processing pipeline for aggregation and MDS.

---

## Troubleshooting

**Images don't appear / sort area is empty**
: Check that `experimental_trials.stimuli_path` and `deployment.debug_shine_variant` (in
`"debug"` mode) in `task_config.json` are correct and that `generate_manifest.py` found your
images. Confirm the web server
is running from the **repo root**, not from `SpAM_Task/` — `index.html` lives at the
root and references task code via `SpAM_Task/...` paths.

**"No participant ID detected" message**
: You accessed the URL without a `PROLIFIC_PID` parameter. Either add `?PROLIFIC_PID=test` to
the URL manually, or set `deployment.mode` to `"debug"` in `task_config.json`.

**Pavlovia shows a blank page or 404**
: Make sure `SpAM_Task/stimuli_manifest.json` is committed to the GitLab repository
(the `*.json` rule in `.gitignore` ignores it by default; the `!SpAM_Task/stimuli_manifest.json`
exception un-ignores it). Verify `index.html` is at the project root on Pavlovia
(Pavlovia auto-detects it there and cannot be configured to look elsewhere).

**Catch trial Done button never enables**
: `catch_trials.location_tolerance` may be too strict for your screen-size range —
`attachCatchCompliance` blocks submission until every image is within tolerance of the target.
Try increasing it (e.g. 0.20-0.30), then re-run a pilot.

**Participant sees the same images as a previous participant**
: This would only happen if two participants share a Prolific PID. Prolific guarantees unique
PIDs — if you are testing with the same PID repeatedly, the stimulus assignment will always be
identical by design (reproducibility). Use different values of `?PROLIFIC_PID=` for each test
run.
