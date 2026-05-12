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

The task expects three sets of PNG images. Create a directory for each (they can live anywhere
on your machine; you will point `config.json` at them in Step 2).

### Main stimuli
Your experimental image library — the images you want similarity data for. There is no hard
limit on how many images you can have; the task samples a subset per participant so that the
full library is covered across participants. Images should:
- Be PNG format with a transparent or uniform background
- Be similar in size (the task renders them all at the same display size, so wildly different
  aspect ratios will look odd)

### Practice images
A small set (~8) of familiar, everyday objects used in the unrecorded warm-up trial. These
should be simple and unambiguous — participants use them to learn the drag interface before
real data collection begins. **These images are not part of your dataset** and are discarded.

### Catch images
A separate set of images (~20 or more) used to check whether participants are following
instructions. On catch trials, participants are told to place all images in a specific region
of the screen (e.g. "top left corner"). Responses that ignore the instruction are flagged.
Use images that are distinctive enough that participants have no reason to be confused — simple
emoji-style or clip-art images work well.

---

## Step 2 — Configure the experiment

Open `SpAM_Task/config.json` in a text editor. The parameters you are most likely to need to
change are:

```json
{
  "stimuli_paths": {
    "main":     "path/to/your/main/images",
    "practice": "path/to/your/practice/images",
    "catch":    "path/to/your/catch/images"
  },
  "deployment": {
    "prolific_completion_url": ""
  }
}
```

**Paths** must be relative to `SpAM_Task/` and use forward slashes — they serve as both
filesystem paths (for `generate_manifest.py`) and URL prefixes (for the browser, which can
only load files within the HTTP server's root directory).

If your images live **outside** `SpAM_Task/`, create a directory junction (Windows) or
symlink (macOS/Linux) inside `SpAM_Task/assets/` pointing at the external directory, then
set the config path to the junction:

```powershell
# Windows PowerShell — run once; no data is copied
New-Item -ItemType Junction `
    -Path  "SpAM_Task\assets\stimuli" `
    -Target "C:\path\to\your\images"
```
```bash
# macOS / Linux
ln -s /path/to/your/images SpAM_Task/assets/stimuli
```

Then set `stimuli_paths.main` to `"./assets/stimuli"` in `config.json`.
The junction is local-only and does not affect deployment.

**Leave `prolific_completion_url` empty for now** — you will fill it in after creating your
Prolific study (Step 6).

### Tuning the trial design

The defaults below work for a dataset of ~750 images with 50–75 participants. If your dataset
is substantially smaller or larger, adjust accordingly.

| Parameter | Default | What it controls |
|---|---|---|
| `design.trials_per_subject` | 10 | How many sorting trials each participant completes |
| `design.images_per_trial` | 20 | How many images appear on screen per trial |
| `design.unique_images_per_subject` | 150 | How many distinct images each participant sees |
| `catch_trials.num_trials` | 2 | How many attention-check trials are interleaved |
| `catch_trials.images_per_trial` | 10 | Images per catch trial |

> **Coverage check**: with your chosen parameters, verify that each image will be seen by
> enough participants for reliable averaging.
> Expected views per image ≈ `N × unique_images_per_subject / total_images`.
> Aim for at least 10–15 views per image.

### Quality control thresholds

| Parameter | Default | Meaning |
|---|---|---|
| `quality_control.min_trial_rt_ms` | 5000 | Trials completed in under 5 s are flagged as too fast |
| `quality_control.min_pairwise_distance_sd` | 0.04 | Main trials where all images are piled in one spot are flagged |
| `catch_trials.cluster_max_mean` | 0.15 | Catch trial: images must be clustered (low mean distance) |
| `catch_trials.cluster_max_sd` | 0.10 | Catch trial: cluster must be tight (low SD) |
| `catch_trials.location_tolerance` | 0.20 | Catch trial: cluster centre must be near the instructed region |

A participant is excluded in post-processing if more than 30% of their trials are flagged.
You can leave these at their defaults for a pilot run and revisit after inspecting the data.

---

## Step 3 — Generate the stimulus manifest

The browser cannot list the contents of a server directory, so the task needs a pre-built
index of your images. Generate it by running:

```bash
cd SpAM_Task
python generate_manifest.py
```

This reads your paths from `config.json`, scans each directory recursively for `.png` files,
and writes `stimuli_manifest.json`. Run it again any time you add, remove, or rename images.

You should see output like:
```
images: 754 images in /path/to/your/main/images
practice_images: 8 images in /path/to/your/practice/images
catch_images: 22 images in /path/to/your/catch/images

Manifest written to .../SpAM_Task/stimuli_manifest.json
```

If any warnings appear (missing path, too few images), fix them before continuing.

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

1. Enable debug mode in `config.json` (bypasses the Prolific ID requirement):
   ```json
   "debug": true
   ```

2. Start a local web server from the `SpAM_Task/` directory:
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

5. When you are satisfied, set `"debug": false` before deploying.

---

## Step 6 — Deploy to Pavlovia

1. **Create a Pavlovia project**: log in to [pavlovia.org](https://pavlovia.org), create a
   new project, and push your repository (including `SpAM_Task/` and all `jspsych/` files) to
   its GitLab repository. The `stimuli_manifest.json` file must be committed — it is
   gitignored by default in this repo, so either remove it from `.gitignore` or commit it
   explicitly (`git add -f SpAM_Task/stimuli_manifest.json`).

2. **Set the experiment URL**: in Pavlovia's dashboard, set the experiment entry point to
   `SpAM_Task/index.html`.

3. **Activate the experiment**: switch the project status to *Running*.

4. **Copy the Pavlovia experiment URL** — it will look like:
   ```
   https://run.pavlovia.org/<your-username>/<project-name>/?PROLIFIC_PID={{%PROLIFIC_PID%}}
   ```
   The `{{%PROLIFIC_PID%}}` part is a Prolific template variable — leave it exactly as shown.

---

## Step 7 — Set up Prolific recruitment

1. Log in to [prolific.com](https://www.prolific.com) and create a new study.
2. Paste the Pavlovia URL (from Step 6) into the *Study URL* field.
3. Under *Completion*, select *I'll redirect them using a URL* and copy the Prolific
   completion URL (it looks like `https://app.prolific.com/submissions/complete?cc=XXXXXXXX`).
4. Paste that completion URL into `config.json`:
   ```json
   "prolific_completion_url": "https://app.prolific.com/submissions/complete?cc=XXXXXXXX"
   ```
5. Commit and push the updated `config.json` to Pavlovia, then re-activate the experiment.

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
: Check that `stimuli_paths.main` in `config.json` is correct and that `generate_manifest.py`
found your images. Confirm the web server is running from `SpAM_Task/`, not from the repo root.
If your images live outside `SpAM_Task/`, the HTTP server cannot reach them — create a
directory junction or symlink as described in Step 2.

**"No participant ID detected" message**
: You accessed the URL without a `PROLIFIC_PID` parameter. Either add `?PROLIFIC_PID=test` to
the URL manually, or set `"debug": true` in `config.json`.

**Pavlovia shows a blank page or 404**
: Make sure `stimuli_manifest.json` is committed to the GitLab repository (it is gitignored by
default — you must add it explicitly). Also verify that the entry point is set to
`SpAM_Task/index.html`, not `index.html` at the repo root.

**Catch trial QC flags everyone**
: The catch-trial tolerance parameters may be too strict for your screen-size range. Try
increasing `catch_location_tolerance` (e.g. 0.25) and `catch_cluster_max_mean` (e.g. 0.20),
then re-run a pilot.

**Participant sees the same images as a previous participant**
: This would only happen if two participants share a Prolific PID. Prolific guarantees unique
PIDs — if you are testing with the same PID repeatedly, the stimulus assignment will always be
identical by design (reproducibility). Use different values of `?PROLIFIC_PID=` for each test
run.
