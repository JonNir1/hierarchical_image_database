"""Minimal end-to-end example of the reusable SpAM simulation pipeline.

Run a new study by editing the two config objects below; no notebook surgery required:

    python -m SpAM_Simulations.example_pipeline            # from the repo root

It demonstrates the full flow - generate a simulation, compute coverage and pre-MDS
reliability, run the (resumable, optionally parallel) MDS sweep into a compact on-disk
ResultStore, then compute post-MDS embedding stability. The MDS sweep needs R + smacof;
everything up to it runs without R.

The heavy ``evaluation.ipynb`` keeps only plotting - it can call these same functions in
place of its compute cells.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from SpAM_Simulations.config import SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline


def main(store_dir: str | Path | None = None, parallel: bool = False) -> None:
    # 1. Describe the simulation. Use (n_images, n_dims) for a random ground truth, or pass
    #    `gt_embeddings=<N x D array>` to drive it from real image features instead.
    sim_config = SimulationConfig(
        n_images=60,
        n_dims=8,
        num_subjects=[20, 40],
        trials_per_subject=[10],
        images_per_trial=[16],
        subjects_noise_scale=[0.0, 0.5],   # relative to gt_distances.std()
        subjects_noise_df=[1],
        reps=3,
        seed=42,
    )

    # 2. Generate (bit-exact serial path; fully reproducible from the seed).
    sim = pipeline.generate_simulation(sim_config, verbose=True)

    # 3. Pre-MDS diagnostics (no R needed).
    coverage = pipeline.compute_coverage_table(sim)
    stability = pipeline.compute_stability_table(sim)
    print(f"\ncoverage rows: {len(coverage)}; mean pair coverage: {coverage['pair_coverage'].mean():.1f}%")
    print(f"pre-MDS reliability (mean Spearman): {stability['spearman'].mean():.3f}")

    # 4. MDS sweep across target dimensions, streamed to a compact ResultStore (needs R).
    sweep_config = MDSSweepConfig(ndims=[6, 8], max_iters=300, convergence_tol=1e-5, precalc_init=False)
    store_dir = Path(store_dir) if store_dir is not None else Path(tempfile.mkdtemp()) / "mds_store"
    store = pipeline.run_mds_sweep(sim, sweep_config, store_dir, parallel=parallel, verbose=True)

    # 5. Post-MDS embedding stability across repetitions.
    embedding_stability = pipeline.compute_embedding_stability(store)
    print(f"\nMDS results stored at: {store_dir}")
    print(f"sweep records: {len(store)}")
    print("post-MDS embedding stability (mean Spearman by config/ndim):")
    print(embedding_stability[["num_subjects", "subjects_noise_scale", "ndim",
                               "n_reps", "mean_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
