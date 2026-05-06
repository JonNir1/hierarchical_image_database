# CLAUDE.md — SpAM_Simulations

Validates whether a given SpAM experimental configuration (subjects, trials, images per trial, noise level) can reliably recover a known ground-truth embedding via MDS. `evaluation.ipynb` is the canonical entry point for the full pipeline.

## Data Flow

1. `Simulation.make(N, D, seed)` — generates N random embeddings in D-dimensional Euclidean space as ground truth; stores pairwise condensed distances.
2. `simulate_experiment(params, gt_distances, rng)` — per subject: draws individual noise from a scaled half-t distribution, then per trial: samples a random image subset and adds Gaussian noise to their pairwise distances. Returns `ExperimentResults` with summed observations and observation counts (NaN for unobserved pairs).
3. `metrics.coverage()` / `metrics.spearman_correlation()` — evaluate data quality before MDS.
4. `run_mds(dists, weights, ndim)` — wraps R's `smacof::mds()` via rpy2 for weighted metric MDS. Weights are `(num_obs > 0)` (binary); missing pairs get weight 0. Optionally bootstraps initial configuration with sklearn's non-metric MDS for faster convergence. Returns a dict with `conf` (embeddings), `confdist` (reconstructed distances), `stress`, `niter`.

## Key Conventions

- All distance vectors use **condensed form** (scipy `pdist` output, length N(N-1)/2). `helpers.convert_to_condensed()` normalizes square↔condensed; call it when the input format is uncertain.
- Subject noise is drawn as `scale * gt_distances.std() * |t(df)| / mean(|t(df)|)` — relative to the GT spread, not absolute.
- `Simulation._results` is a `Dict[ExperimentParameters, List[ExperimentResults]]` — multiple reps per parameter set.
- MDS results are streamed to `mds_results.pkl` by **appending** `(key, value)` pairs with `pkl.dump(..., "ab")`, not standard pickle. Use `load_mds_results()` from the notebook to deserialize this append-log format.
