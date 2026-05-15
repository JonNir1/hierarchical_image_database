# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cognitive neuroscience research project building a hierarchical image dataset (754 PNG images of objects with white/transparent backgrounds). The goal is to collect human perceptual dissimilarity scores via an online Spatial Arrangement Method (SpAM) task, then use Multidimensional Scaling (MDS) to embed them into a perceptual similarity space and compare against semantic hierarchy and pixel-wise similarity.

**Status**: Simulations complete (`SpAM_Simulations/`); dataset visualization tools complete (`visualize_dataset/`); online SpAM task implemented (`SpAM_Task/`), awaiting IRB approval and OSF pre-reg submission before launch. The full analysis plan is drafted in the OSF pre-registration (see `README.md` for a short summary).

## Dependencies

No `requirements.txt` exists. Key Python packages: `numpy`, `scipy`, `scikit-learn`, `pandas`, `plotly`, `tqdm`, `opencv-python` (cv2), `ete3`, `PyQt5`, `rpy2`.

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
