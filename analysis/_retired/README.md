# Retired analyses

Work that was finished or superseded but is kept because its results or code are still cited.
Each directory has a `KEY_FINDINGS.md` (what was found, what was re-run, caveats).

| Directory | What it is | Needs |
|---|---|---|
| `pilot/` | SpAM pilot descriptives across task versions v1 to v4 (`analysis.ipynb`, `figures.py`, `show_trials.py`) | `data/` |
| `image_processing/` | Resize and SHINE_color pipeline (MATLAB) plus `postprocess_analysis.ipynb`, a check that SHINE output is category-neutral | MATLAB with the Image Processing Toolbox for the `.m` files; `images/` for the notebook |
| `rsatoolbox_eval/` | R `smacof` vs `rsatoolbox` MDS backend evaluation (`REPORT.md` is the long form) | R, `smacof`, `rpy2`; built RDMs in `analysis/results/rdms/` |
| `visualize_dataset/` | Category hierarchy and subcategory-size description of the image datasets | `ete3` and `PyQt5` for the tree parts; the Auerbach-Asch and PiCS folders for the external-dataset sections |

## Conventions

- Run notebooks from their own directory. Each starts with a setup cell that finds the repo root
  from the working directory and puts it on `sys.path`; imports are absolute from the repo root.
- Run scripts from the repo root as modules, e.g.
  `python -m analysis._retired.rsatoolbox_eval.mds_rsatoolbox_compare`.
- Live code does not import from here. Shared pieces that outlived their analysis live in
  `analysis/utils/` (for example `null_distances.py`, taken from `pilot/`).
- `ete3`, `PyQt5` and MATLAB are not in `requirements.txt` and not installed in CI.
- The MATLAB tests (`image_processing/__tests__/TestShineEqualize.m`) run in CI on the self-hosted
  `matlab` runner; none of the Python code here has tests of its own.
