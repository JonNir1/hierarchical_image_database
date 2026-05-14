# hierarchical_image_database
Building a database of real-world objects in a neutral background that can be used for neuroscientific studies, along with a concept hierarchy for the objects and similarity scores for image pairs

## Research Questions

1. **Hierarchy validation** — does the curated 754-image set exhibit hierarchical perceptual structure that resembles the Kiani-Mur (2007, 2013) hierarchy?
2. **SHINE effect** — does SHINE-color preprocessing (required for EEG compatibility) preserve that perceptual structure or distort it?
3. **Decomposition shift** — when perceptual distance is modelled as a weighted combination of semantic and sensory distances, does SHINE selectively reduce the sensory weight while leaving the semantic weight intact?

## Analysis Plan (summary)

- Collect SpAM data via Prolific/Pavlovia for two between-subject cohorts (N=75 each; one pre-SHINE, one post-SHINE).
- Build population RDMs per cohort via weighted metric MDS (R `smacof` via `rpy2`).
- **RQ1a (any hierarchy?)** — normalised Gromov δ-hyperbolicity; compared against semantic (tree-like) and sensory (Euclidean) baselines.
- **RQ1b (matches Kiani-Mur?)** — Spearman/Mantel ρ between perceptual RDM and a Kiani-Mur categorical-distance RDM; stratified by tree level.
- **RQ2a (SHINE preserves vs perturbs)** — Spearman ρ(pre, post) tested against (i) an image-label shuffle null for shared structure and (ii) a cross-subject shuffle null for SHINE-specific perturbation.
- **RQ2b** — level-stratified Spearman ρ(pre, post); level × condition interaction.
- **RQ2c** — bootstrap CI on [ρ(pre, KM) − ρ(post, KM)].
- **RQ3 (decomposition)** — fit `D_perc = α·D_sem + β·D_sens` per cohort; bootstrap CIs on (α_pre − α_post) and (β_pre − β_post), under both Kiani-Mur and WordNet `D_sem`.
- Multiple comparisons via FDR within each RQ family.
- Exploratory: hyperbolic-space replication; item-level Procrustes; alternate sensory metrics.

Full pre-registration: *[OSF link, populated after submission]*.
