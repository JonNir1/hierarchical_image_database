# SpAM_Simulations

Simulate SpAM experiments (noisy subject distance judgements over a ground-truth embedding)
and reconstruct them with weighted MDS, to evaluate how well MDS recovers the latent space
under different sampling/noise regimes.

## Modules

| Module | Responsibility |
|---|---|
| `experiment.py` | Core simulation: `simulate_experiment` / `simulate_single_subject` (vectorized, condensed form). |
| `simulation.py` | `Simulation` container + ground-truth distances; `make` (random) / `from_embeddings` (real data). |
| `metrics.py` | `coverage`, `spearman_correlation`. |
| `helpers.py` | Distance-matrix format conversion (`convert_to_condensed`). |
| `multi_dimensional_scaling.py` | `run_mds` - weighted SMACOF via R's `smacof` (needs R + rpy2). |
| `config.py` | `SimulationConfig`, `MDSSweepConfig` - declarative study configuration. |
| `pipeline.py` | Reusable orchestration (generate / coverage / stability / MDS sweep / embedding stability). |
| `storage.py` | `ResultStore` - compact, streamable, resumable on-disk store for sweep results. |
| `example_pipeline.py` | Minimal runnable end-to-end example. |
| `evaluation.ipynb` | Plotting / analysis notebook (can call the pipeline functions in its compute cells). |

## Quick start

```python
from SpAM_Simulations.config import SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline

cfg = SimulationConfig(
    n_images=60, n_dims=8,                       # or: gt_embeddings=<N x D array> for real data
    num_subjects=[20, 40], trials_per_subject=[10], images_per_trial=[16],
    subjects_noise_scale=[0.0, 0.5], subjects_noise_df=[1], reps=3, seed=42,
)
sim = pipeline.generate_simulation(cfg)                      # bit-exact, reproducible from seed
coverage  = pipeline.compute_coverage_table(sim)            # no R needed
stability = pipeline.compute_stability_table(sim)           # no R needed

sweep = MDSSweepConfig(ndims=[6, 8], max_iters=300, precalc_init=False)
store = pipeline.run_mds_sweep(sim, sweep, "mds_store", parallel=False)   # needs R; resumable
emb   = pipeline.compute_embedding_stability(store)
```

Or run the bundled example: `python -m SpAM_Simulations.example_pipeline` (from the repo root).

### Performance / storage notes
- Generation is vectorized and runs in condensed form (~9-10x faster than the original loop,
  ~half the memory). The serial path is bit-exact and fully reproducible from `seed`.
- `run_mds_sweep(parallel=True, n_jobs=...)` distributes the independent MDS runs across
  processes (joblib/loky), streaming results to disk so peak memory stays bounded.
- `ResultStore` keeps a human-readable `meta.csv` plus a flat float32 `confdists.f32`
  (memory-mapped on read), replacing the old multi-GB append-only pickle.

## Running with R (rpy2 + smacof)

`multi_dimensional_scaling.py` imports R at load time. R 4.5 + the `smacof` package are
required. On Windows without Rtools, `R CMD config` cannot run; rpy2 must fall back to the
DLL in `R_HOME/bin/x64`. That fallback only triggers when the `config` subprocess fails
*cleanly*, which it does from a normal Windows shell but **not** from Git Bash (where `sh`
makes the config script exit 0 with empty output and crashes rpy2's parser).

Working setup used for this project:

```
R_HOME       = C:\Program Files\R\R-4.5.2
R_LIBS_USER  = C:\Users\nirjo\R_library\4.5      # where smacof is installed
```

From a plain PowerShell/cmd session (with the venv active) rpy2 initialises fine. If you must
run it under Git Bash, give the Python process a Windows-only PATH (no Unix `sh`), e.g.
`PATH="/c/Program Files/R/R-4.5.2/bin/x64:/c/Windows/System32:/c/Windows"`, so `R CMD config`
fails fast and rpy2 falls back to `bin/x64`.

`multi_dimensional_scaling.py` automatically prepends `R_HOME\bin\x64` to PATH at import time, so
loading R packages (whose DLLs depend on `R.dll`/BLAS there) works without you editing PATH. If R
still isn't found, set `R_HOME` explicitly.

## Tests

```
.venv/Scripts/python.exe -m pytest SpAM_Simulations/__tests__ -q
```

R-dependent tests (`test_pipeline_mds.py`) auto-skip if the R bridge can't be imported; the
rest run anywhere. With R configured the full suite is 55 tests.
