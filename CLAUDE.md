# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive neuroscience research project building a hierarchical image dataset (725 curated PNG images of objects with white/transparent backgrounds, distilled from six source datasets — see `source_datasets/`). The goal is to collect human perceptual dissimilarity scores via an online Spatial Arrangement Method (SpAM) task, then use Multidimensional Scaling (MDS) to embed them into a perceptual similarity space and compare against semantic hierarchy and pixel-wise similarity.

**Status**: Pilot data collected (`data/pilot/`); simulations complete (`SpAM_Simulations/`); dataset visualization tools complete (`visualize_dataset/`); online SpAM task implemented (`SpAM_Task/`). The full analysis plan is drafted in the OSF pre-registration (see `README.md` for a short summary).

## Data Policy

**`data/` is gitignored and must NEVER be committed or pushed to any remote** (GitHub `origin` or Pavlovia `pavlovia`). It contains human subjects data (pilot and full study SpAM responses). Keep it local only. The `.gitignore` entry covers all subdirs including `data/pilot/`.

## Environment

**Python virtual environment**: `.venv/` at the repo root (do NOT create a new venv or
install packages system-wide). Activate with `.venv\Scripts\activate` (Windows) or
`source .venv/bin/activate` (bash). Use `.venv/Scripts/python.exe` for direct invocation.

Install / update all dependencies from the repo root:

```bash
.venv/Scripts/pip install -r requirements.txt
```

**`requirements.txt`** at the repo root lists all Python dependencies. Add new packages
there rather than installing ad hoc.

R dependency (required for MDS): `smacof` — install in R with `install.packages("smacof")`. The `rpy2` bridge is loaded at import time in `SpAM_Simulations/multi_dimensional_scaling.py`, so importing that module without R/smacof configured will raise immediately.

## Running Things

Modules use the project root as the package root (no `__init__.py` files), so run scripts from the repo root:

```bash
jupyter notebook SpAM_Simulations/evaluation.ipynb
jupyter notebook visualize_dataset/dataset_visualizations.ipynb
```

The SpAM task is served from the repo root via `index.html` (a thin entry point that
loads scripts from `SpAM_Task/`). For local browser testing: `python -m http.server`
from the repo root, then open `http://localhost:8000/`.

There is a test suite under `SpAM_Task/__tests__/` (pytest for the Python manifest
generator, `node --test` for the JS modules).
