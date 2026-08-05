# OSF Pre-Registration Draft

> **STATUS: WORK IN PROGRESS — NOT YET REGISTERED.**
> This is the working draft. The OSF-registered version (once submitted) will be
> tagged in this repo at `osf-submission-v1` and any post-submission methodological
> changes go into a separate amendments file, never edited into this draft after
> the OSF timestamp.

Structured to the [OSF Standard Pre-Registration](https://docs.google.com/document/d/1gkN0Jp6Gu7GIA4Ne4YCDZ61nCLQRgt32moRdUg9AnVg/edit?tab=t.0)
template. Section headings match the  OSF form.

---

## Metadata
### Title
The effects of image preprocessing on the semantic-sensory decomposition of perceptual similarity in a
curated hierarchical single-object image dataset.

### Description
Perceptual similarity analysis of a curated 725-image dataset, with and without low-level image preprocessing.<br>
Our goal is to examine the effect of a commonly used image preprocessing pipeline on the geometric structure of 
human perceptual similarity, and its decomposition into _semantic_ and _sensory_ components.
<br><br>
The dataset consists of 725 images of real-world objects on a neutral background, curated from the dataset used by [Auerbach-Asch et al., 2023](https://doi.org/10.1101/2023.06.28.546397). The current dataset is organized in a hierarchical directory 
structure that encodes semantic hierarchy for the objects, following a well-established hierarchy (see 
[Kiani et al., 2007](https://journals.physiology.org/doi/full/10.1152/jn.00024.2007), 
[Kriegeskorte et al., 2008b](https://www.cell.com/neuron/fulltext/S0896-6273(08)00943-4), 
[Mur et al., 2013](https://doi.org/10.3389/fpsyg.2013.00128), 
[Cichy et al., 2014](https://www.nature.com/articles/nn.3635), 
[Grootwagers et al., 2018](https://www.sciencedirect.com/science/article/abs/pii/S1053811918305299),
and [Auerbach-Asch et al., 2023](https://doi.org/10.1101/2023.06.28.546397)).

Note that this dataset is an extension of the curated 92-image set used in Kriegeskorte (2008b), Mur (2013), 
and Cichy (2014), which posed a potential threat for "overfitting" to a single dataset ([Grootswagers & Robinson,
2021](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.682661/full)).
<br><br>
The dataset includes two variants of each image:
- the original images after background removal and resizing to 170x170 pixels.
- the same images after `SHINE-color` ([Dal Ben, 2023](https://doi.org/10.1016/j.mex.2023.102377)) preprocessing, which normalizes low-level image 
  properties (luminance, color histogram) while other low-level features information. This preprocessing is 
  commonly used to ensure EEG compatibility in visual experiments, but its effect on perceptual similarity 
  structure is not well understood.
<br>
Perceptual similarity scores are collected using the Spatial Arrangement Method (SpAM; [Hout et al., 
2013](https://doi.org/10.3758/s13428-012-0265-0), [Mur et al., 2013](https://doi.org/10.3389/fpsyg.2013.00128)) 
via a web-based task deployed on Pavlovia.<br>
Participants are randomly assigned to either the pre- or post-SHINE cohorts, with a between-subject design. 
We will build representational dissimilarity matrices (RDMs) from the SpAM data and test a set of hypotheses 
about the geometric structure of perceptual similarity and its relationship to semantic and sensory similarity, 
as well as the effect of SHINE preprocessing on that structure and decomposition.

### Authors

Jonathan Nir [1]  
Leon Y. Deouell [1, 2]  

[1] Edmond and Lily Safra Center for Brain Sciences, The Hebrew University of Jerusalem  
[2] Department of Psychology, The Hebrew University of Jerusalem

## Overview
### Research questions or hypotheses

**RQ1:** Does the curated image dataset exhibit hierarchical perceptual structure that resembles the
Kiani-Mur (2007, 2013) hierarchy?

H1a: The pre-SHINE perceptual RDM is more hierarchical ("tree-like") than a pixel-wise sensory RDM.  
We measure "tree-like" using the [Gromov δ-hyperbolicity metric](https://en.wikipedia.org/wiki/Hyperbolic_metric_space),
so this translates to $$H1a: δ̂(D^{pre}_{perc}) < δ̂(D^{pre}_{sens})$$
H1b: The pre-SHINE perceptual RDM matches the Kiani-Mur hierarchy. We first generate a Kiani-Mur based
semantic-distance RDM, where distance is defined as the number of edges in the Kiani-Mur graph between two images' categories.
We then test whether Spearman ρ between $D^{pre}_{perc}$ and $D^{pre}_{KM}$ is significantly greater than zero, with the effect
strongest at superordinate levels and weakest at basic level (Mur 2013): $$H1b: ρ(D^{pre}_{perc}, D^{pre}_{KM}) > 0$$
<br>
*Exploratory Analyses for RQ1:*  
H1c: Repeat the analysis from H1b using WordNet-based semantic distances (synsets assigned from image category labels) instead of KM graph distances.
The hypothesis is similar: $$H1c: ρ(D^{pre}_{perc}, D^{pre}_{WN}) > 0$$

**RQ2:** Does SHINE-color preprocessing preserve that perceptual structure or distort it?

H2a: Shared structure exists between cohorts. Spearman ρ between $D^{pre}_{perc}$ and $D^{post}_{perc}$ is
significantly greater than zero against an image-label shuffle null: $$H2a: ρ(D^{pre}_{perc}, D^{post}_{perc}) > 0$$
H2b: SHINE perturbs structure-similarity beyond a random split. Spearman ρ between $D^{pre}_{perc}$ and $D^{post}_{perc}$
is lower than the distribution of ρ values produced by random subject reassignment to fake pre/post cohorts
(10000-iteration cross-subject shuffle null):
$$H2b: ρ(D^{pre}_{perc}, D^{post}_{perc}) < ρ_{null}(D^{pre*}_{perc}, D^{post*}_{perc})$$

**RQ2 is determined by the combined results of H2a & H2b:**

| H2a rejects? | H2b rejects? | Verdict      | Interpretation                                |
|--------------|--------------|--------------|-----------------------------------------------|
| yes          | no           | preservation | cohorts agree as much as a random subject-split would predict (no SHINE-driven gap) |
| yes          | yes          | perturbation | shared structure remains, but SHINE shifted it|
| no           | yes          | destruction  | SHINE eliminated shared image-specific structure |
| no           | no           | inconclusive | underpowered or pathological data            |

<br><br>
*Exploratory Analyses for RQ2:*  
H2c: SHINE degrades the Kiani-Mur match. We test whether the lower bound of the bootstrap CI of the difference
$ρ(D^{pre}_{perc}, D_{KM}) − ρ(D^{post}_{perc}, D_{KM})$ is greater than zero (i.e., pre-SHINE matches Kiani-Mur
more strongly than post-SHINE).  
H2d: SHINE effect is level-stratified. We compute per-level Spearman ρ between $D^{pre}_{perc}$ and $D^{post}_{perc}$, and test for
a significant `level × condition` interaction, with the prediction that ρ is higher at superordinate levels than at
basic level.

**RQ3:** When perceptual distance is modelled as a weighted combination of semantic and sensory distances, does SHINE
selectively reduce the sensory weight while leaving the semantic weight intact?

H3 (confirmatory, 3-predictor): In the model
$$D_{perc} = α·D_{sem} + β·D_{CLIP} + γ·D_{sens} + ε$$
fit per cohort via non-negative least squares (NNLS), with main analysis using
$D_{sem} = D_{KM}$ (Kiani-Mur graph distance):

(i)  $γ^{pre} > γ^{post}$ — SHINE reduces the pure-sensory (low-level pixel)
     contribution. Operationally: lower bound of $CI(γ^{pre} − γ^{post}) > 0$.
(ii) $α^{pre} ≈ α^{post}$ — SHINE preserves the pure-semantic contribution.
     Operationally: $0 \in CI(α^{pre} − α^{post})$.

Reported but not directionally pre-registered:
(iii) $β^{pre}$ vs $β^{post}$ — change in the visual-semantic (CLIP) contribution.
      Bootstrap CI reported without a directional claim, because the impoverished
      single-object stimulus set may or may not show a unique CLIP component
      analogous to Shoham et al. (2024).

CIs throughout: 5k bootstrap resamples over subjects, 95% percentile.

Additional reporting (following Shoham et al. 2024, Nature Human Behaviour):
- **Hierarchical multiple linear regression**: predictors entered in fixed order
  $D_{sens} \to {+}D_{CLIP} \to {+}D_{sem}$; report the additional $R^2$
  contributed by each predictor when it enters the model.
- **Partial correlations**: each predictor while holding the other two constant
  (unique-variance contribution). Fisher-z transform on correlations before any
  t-tests. FDR (Benjamini-Hochberg) across the three predictors.

The same confirmatory model is also fit under $D_{sem} = D_{WN}$ (WordNet shortest-path
distance, synsets from image category labels) as a registered supplementary robustness check; agreement between
sources is informative but not required for the H3 conclusion.

H3 exploratory variants (registered, not confirmatory):
- $D_{sem} = D_{sem}^{SGPT}$ — matches Shoham et al.'s exact "semantic"
  specification.
- 4-predictor model $D_{perc} = α·D_{sem} + β·D_{CLIP} + γ·D_{VGG} + δ·D_{pix} + ε$,
  decomposing the visual contribution into high-level (VGG FC7) and low-level
  (pixel) components. Tests the sharper prediction that SHINE selectively reduces
  $δ$ (pixel) while preserving $γ$ (VGG), $β$ (CLIP), and $α$ (semantic).
  Reported descriptively only — no FDR-corrected directional claims on the
  individual coefficients in this expanded model.

*Note on predictor collinearity*: Shoham et al. (2024) hand-picked 20 stimulus
objects to minimize the correlation between their visual (VGG), visual-semantic
(CLIP), and semantic (SGPT) DNN embeddings, giving them near-orthogonal predictors.
Our 725-image set was curated for hierarchical coverage of Kiani-Mur, with no such
orthogonality guarantee. Predictor multicollinearity in our H3 regression will
therefore be substantively higher than in Shoham et al. NNLS coefficients on
correlated predictors are known to be unstable; the partial-correlation and
hierarchical-$R^2$ analyses in the reporting protocol mitigate this by attributing
unique vs shared variance per predictor.
<br><br><br>
**Exploratory / Supplementary Analyses; not part of main RQs:**
- use hyperbolic embeddings (e.g. Poincaré disk; Marton 2025 HyPoE or equivalent) to replicate the H2 and H3 analyses in a non-Euclidean space.
- item-level Procrustes displacement maps to visualize which images move the most between pre- and post-SHINE spaces, and whether those movements are consistent with a sensory vs semantic shift (e.g. top-K=5 nearest-neighbour Jaccard).
- alternate sensory metrics: Gabor-bank distance; corneal/V1-filtered pixel distance; VGG-16 FC7 high-level visual features (used as the 4th predictor in the exploratory 4-predictor H3 variant).
- alternate semantic metric: $D_{sem}^{SGPT}$ (SGPT bi-encoder sentence embeddings of per-image descriptions), matching Shoham et al. 2024's exact "semantic" operationalization.

### Foreknowledge of data or evidence
Data does not yet exist. No part of the data that will be used for this analysis plan exists, and no part will be generated until after this plan is registered.

## Research Design
### Study type
+ Quasi-experimental design: The strategy to identify or estimate a causal relationship takes advantage of some particular feature(s) or circumstance(s) that helps avoid much of the need for adjustment or control variables. This may include instrumental variables, "natural" experiments, interrupted time series designs, difference-in-difference designs, synthetic controls, or related approaches.
+ Other:  
  Two between-subject cohorts (pre-SHINE and post-SHINE), recruited concurrently via a single Prolific study.
  Cohort assignment is determined automatically at task startup by a deterministic function of the participant's
  Prolific ID (see *Randomization* below), so both cohorts are collected in parallel.

### Intention for causal interpretation (optional)
Direct inference on causal relationship(s): This study is intended to infer or estimate a causal relationship between two or more variables. It is designed specifically for the purposes of causal inference or identification

### Blinding of experimental treatments
Subjects will not be aware of the assigned treatment during data collection (either because the subjects are not human participants or because of blinding procedures).

### Study design
Two between-subject cohorts: pre-SHINE and post-SHINE.  
Cohort-level RDMs will be generated and compared to each other and to reference (semantic/sensory) RDMs.

### Randomization (optional)
Each participant is assigned to a cohort automatically at task startup by `hashString(PID) % 2`,
where `hashString` is the djb2 hash of the Prolific ID: even hash value → pre-SHINE cohort,
odd hash value → post-SHINE cohort. The assignment is deterministic and reproducible from the
Prolific ID, and is recorded in every saved trial row (`shine_variant` field).
This produces an approximately equal split across participants.  
A subset of `M=150` images from the 725 dataset is assigned to subjects based on their Prolific-ID (RNG seed).  
Of those, each subject performs `t=10` trials, each with a randomly assigned subset of `k=20` images.  
A subset of `n_double=50` images is repeated across two different trials for within-subject reliability estimation.  
Catch-trial positions and target locations are drawn from the same RNG.

## Sampling
### Data collection procedures
Subjects will be recruited via Prolific, and will perform the SpAM task hosted on Pavlovia.  
Each subject will be assigned to either the pre-SHINE or post-SHINE cohort, and will perform the SpAM task on a subset
of images from the dataset (randomly assigned per subject).

### Sample size
We will retain a minimum of `N=75` subjects per cohort (`150` total), excluding those who fail the
quality control trials. Sampling will continue until each cohort has 75 subjects and its dissimilarity graph is
strongly connected.  
Split-half Spearman ρ per cohort will be computed and reported as a diagnostic; it will not affect stopping.

### Sample size rationale (optional)
A simulation of the SpAM+MDS pipeline ([code is available here](https://github.com/JonNir1/hierarchical_image_database/tree/main/SpAM_Simulations))
with `t=10` trials × `k=20` images/trial, ground-truth dim `D=10`, SNr=0.8 shows:
- dissimilarity graph is strongly connected at `N ≥ 50`
- MDS was able to converge with target dimensionality of `d=10` for any `N ≥ 50`
- Even for target dimensionality of `d=5`, MDS showed promising convergence for `N ≥ 50`
- Embedding reliability (comparing two independent samples of N subjects) showed a Spearman `ρ ≈ 0.6` at `N=75` for `d=5`, and even higher if we increase `d` or reduce `SNr`.

### Starting and stopping rules
We will collect a minimum of `N=75` subjects per cohort (excluding those who fail quality control), and will
continue sampling until each cohort has 75 subjects **and** its dissimilarity graph is strongly connected (checked in 
intervals of 5 subjects after reaching N=75).


## Variables
### Manipulated variables
- Image version (pre-SHINE / post-SHINE), between-subject.

### Measured variables
- `moves`: array of `{src, x, y, t}` from jsPsychFreeSort
- `final_locations`: array of `{src, x, y}` from jsPsychFreeSort
- `pairwise_distances`: normalised Euclidean distances
- `pairwise_distance_sd`: SD of normalised distances
- `rt`: trial duration in ms
- `qc_flag`: boolean from main-trial / catch-trial QC
- `within_subject_reliability`: per-subject Spearman ρ between the two distance vectors
  for each of the 50 within-subject-repeated images, across the two trials in which it
  appears. Computed post-hoc from `pairwise_distances`. Used for subject exclusion.


### Indices
- Subject-level reliability index: Spearman ρ on the 50 within-subject repeated images
- Population-level perceptual RDMs from unweighted metric MDS (`smacof` in R; binary pair weights — 1 if observed, 0 otherwise): $D^{pre}_{perc}$ and $D^{post}_{perc}$
- Semantic distance matrices: $D_{sem}^{KM}$ (Kiani-Mur graph distance) and $D_{sem}^{WN}$
  (WordNet shortest-path distance). For $D_{sem}^{WN}$, each image is assigned a WordNet
  noun synset from its filename stem and parent directory label: e.g., `duck3.png` in a
  `bird/` subdirectory resolves to `duck.n.01`; a `bird/` image whose filename does not
  yield a valid synset falls back to `bird.n.01`. Manual overrides handle polysemous stems
  where WordNet's most-frequent sense is incorrect (e.g., `screw` -> `screw.n.04` not
  `prison_guard.n.01`; `lettuce` -> `lettuce.n.02` not `boodle.n.01`). Human-face images
  are assigned `woman.n.01` or `man.n.01` from a gender token in the filename; human-body
  images resolve via their category dirname (e.g., `clown.n.01`, `child.n.01`, `hand.n.01`).
  Synset assignments are stored in `images/manifest.csv` (column `wn_synset_name`);
  the assignment code is in `analysis/rdms/assign_wn_synsets.py`.
  *Note on methodology change*: the original plan was to classify each image with
  ResNet-50 and map the predicted ImageNet class to its WordNet synset. This approach was
  abandoned because ImageNet-1K does not include person, face, or human-body categories,
  leaving 22.5% of images (all 164 human images) unclassifiable, with fallback distances
  that do not reflect semantic content. White/transparent backgrounds also remove contextual
  cues that ResNet relies on. Since all images were manually curated with known category
  labels, direct label-to-synset mapping is both more accurate and fully deterministic.
- Sensory distance matrices: $D_{sens}^{pre}$ and $D_{sens}^{post}$ — pairwise Euclidean
  distance between flattened image vectors, before and after SHINE preprocessing,
  respectively. Captures low-level pixel info that SHINE directly manipulates.
- Visual-semantic distance matrix: $D_{CLIP}^{pre}$ and $D_{CLIP}^{post}$ — pairwise
  cosine distance between OpenAI pretrained CLIP ViT-B/32 output-layer image
  embeddings, computed on pre-SHINE and post-SHINE images respectively. Following
  Shoham et al. (2024, *Nature Human Behaviour*), who showed that CLIP-derived
  RDMs capture variance in human similarity beyond what pure-visual or pure-semantic
  predictors explain.
  *Implementation note on the CLIP variant*: OpenAI trained CLIP with QuickGELU
  activations. `open_clip` (the library used here) subsequently split this into two
  model configs, with plain `ViT-B-32` defaulting to standard GELU and
  `ViT-B-32-quickgelu` retaining the original. We load `ViT-B-32-quickgelu` with
  `pretrained="openai"`, so that the activation function matches the one the released
  weights were trained under. Pairing plain `ViT-B-32` with the OpenAI weights instead
  is a silent config mismatch (`open_clip` emits a load-time warning): across the full
  725-image set (262,450 pairs) it shifted the CLIP RDM by mean |Δd| = 0.023 against a
  mean distance of 0.36, giving Spearman ρ = 0.943 (pre-SHINE) and ρ = 0.938
  (post-SHINE) with the correctly-configured RDM. That is a small but systematic
  distortion, i.e. roughly 6% of the rank structure of the $D_{CLIP}$ predictor would
  have been set by a configuration artifact rather than by the model. We record this
  because "CLIP ViT-B/32" alone does not uniquely identify a network under current
  `open_clip` versions, and the choice is consequential for β in the H3 decomposition.
- (Exploratory) High-level-visual distance matrix: $D_{VGG}^{pre}$ and $D_{VGG}^{post}$
  — pairwise cosine distance between VGG-16 penultimate-layer (FC7) activations on
  pre- and post-SHINE images. Uses ImageNet-pretrained VGG-16 from `torchvision`.
  Matches Shoham et al.'s exact "visual" operationalization; used as a sensitivity-
  analysis predictor representing high-level visual features (in contrast to the
  primary pixel-wise $D_{sens}$ which captures low-level features that SHINE directly
  manipulates).
- (Exploratory) SGPT-based semantic distance: $D_{sem}^{SGPT}$ — pairwise cosine
  distance between SGPT sentence embeddings (1.3B-param GPT bi-encoder, output
  layer) of per-image category descriptions. Variant-agnostic (one RDM, used for
  both cohorts). Registered as a sensitivity check on $D_{sem}^{KM}$, matching
  Shoham et al.'s exact "semantic" operationalization.
- Pre/post image correspondence is established by **filename match**. SHINE preserves
  filenames and directory structure (`images/pre_shine/<cat>/<name>NN.png` ↔
  `images/post_shine/<cat>/<name>NN.png`), so the i-th row/column of $D^{pre}_{sens}$
  corresponds to the same object as the i-th row/column of $D^{post}_{sens}$ (and
  likewise for the perceptual RDMs, $D_{CLIP}$, and $D_{VGG}$).

## Analysis Plan
### Statistical models
- Population RDM: unweighted metric MDS via R package `smacof` (binary pair weights: 1 if pair observed, 0 otherwise)
- Hierarchy validation: Gromov δ-hyperbolicity metric for each RDM (bootstrap sampling of 10⁵ 4-tuples)
- Spearman ρ between perceptual RDMs and sensory/semantic/visual-semantic RDMs
- Decomposition (following Shoham et al. 2024, *Nature Human Behaviour*):
  - **Confirmatory**: NNLS regression on condensed RDMs of the form
    $D_{perc} = α·D_{sem} + β·D_{CLIP} + γ·D_{sens} + ε$. Bootstrap CIs (5k
    subject resamples) on coefficients and pre-post differences.
  - **Variance attribution**: hierarchical multiple linear regression with
    predictors entered in fixed order ($D_{sens} \to {+}D_{CLIP} \to {+}D_{sem}$);
    report $\Delta R^2$ per step. Partial correlations for unique-variance
    contribution of each predictor controlling for the other two. Fisher-z
    transform on correlations before t-tests; FDR (Benjamini-Hochberg) across
    the three predictors.
  - **Exploratory 4-predictor variant** decomposes the visual contribution by
    adding $D_{VGG}$ (VGG-16 FC7) between $D_{sens}$ (pixels) and $D_{CLIP}$,
    probing whether SHINE selectively reduces low-level (pixel) versus
    high-level (VGG) visual contributions.
- Optional: Procrustes analysis for visualizing structural differences 

### Transformations
- pairwise distances are normalized to [0, 1] to account for different screen sizes across participants.

### Inference criteria
- We will use permutation tests to compare the pre/post cohorts
- CIs will be calculated using bootstrapped samples of subjects
- we use a 0.05 threshold for significance and will apply FDR corrections for multiple comparisons
  within each RQ.

### Data inclusion and exclusion
- Trial-level: trials are flagged if they fail quality control checks
- Subject-level: subjects are excluded if ANY of:
  (a) >30% of their main trials are flagged by trial-level QC,
  (b) they fail the catch-trial QC (per-catch-trial thresholds in `task_config.json`),
  (c) their within-subject reliability index is in the lowest decile of their cohort.
  Exclusion criterion (c) is computed per cohort once N=75 is reached; subjects below
  the cohort-specific 10th percentile are excluded and replaced. Recursive: after
  replacing, the cohort's 10th percentile is recomputed and any newly-bottom-decile
  subjects are also excluded, until the bottom-decile cutoff stabilises.
- Excluded subjects are replaced until we reach N=75 retained per cohort.

### Missing data
- missing dissimilarity between pairs is handled natively by `smacof` MDS (binary pair weights — pairs not observed by any subject get weight 0 and are inferred via stress-minimization from the observed pairs).


## Other
- Code availability: https://github.com/JonNir1/hierarchical_image_database/tree/main
- Image sharing: images are part of several published datasets and can be accessed according to their respective
  licenses. We will release a manifest with source attribution for each image. Raw & processed images will not be
  redistributed due to licensing restrictions.
- Data sharing:
  - dataset manifest with source attribution will be shared on OSF
  - raw trial data will not be shared due to privacy and data-use agreements
  - aggregated MDS embeddings, hierarchy labels, and analysis code will be shared on OSF
  

---

## Supporting context (not part of OSF submission)

### Research Questions (refined, full numbering)

**RQ1 — Hierarchy validation.**
- RQ1a: Do we have *any* hierarchy? — Gromov δ-hyperbolicity primary; CCC and additive-tree
  fit secondary. Applied to `D_perc_*` with `D_sem_*` and `D_sens_*` as baselines.
- RQ1b: Does our hierarchy resemble Kiani-Mur? — Spearman/Mantel ρ between $D^{pre}_{perc}$
  and a Kiani-Mur categorical-distance RDM, stratified by level. Procrustes / cophenetic /
  Robinson-Foulds as secondary structural checks.

**RQ2 — SHINE effect on perceptual structure.**
- RQ2a: Global similarity pre vs post. Spearman ρ($D^{pre}_{perc}$, $D^{post}_{perc}$) against TWO
  nulls: image-label shuffle (Claim A: shared structure) AND cross-subject shuffle
  (Claim B: SHINE-driven gap). RQ2a is supported only if BOTH reject.
- RQ2b: Level-stratified SHINE effect — per-level Spearman ρ; level × condition test.
- RQ2c: Does SHINE move structure away from Kiani-Mur? — bootstrap CI on
  [ρ(pre, KM) − ρ(post, KM)] excludes zero in the positive direction.

**RQ3 — Decomposition shift.**
- Model $D_{perc} = α·D_{sem} + β·D_{sens} + ε$, fit per condition. Main analysis uses
  $D_{sem} = D_{KM}$; $D_{sem} = D_{WN}$ is reported as supplementary robustness.
  Hypotheses: $β^{post} < β^{pre}$; $α^{post} ≈ α^{pre}$.

### Must-have vs nice-to-have

**Must-have (confirmatory):**
- Data collection 2 × N=75 between-subject
- RQ1a (any hierarchy)
- RQ2a (global SHINE effect)
- RQ3 (decomposition under at least one D_sem source)
- Dataset deliverable D1

**Nice-to-have (secondary, registered):**
- RQ1b (Kiani-Mur resemblance, level-stratified)
- RQ2b (level-stratified SHINE)
- RQ2c (post-vs-KM compared to pre-vs-KM)
- RQ3 under WordNet D_sem (robustness)

**Exploratory (not confirmatory):**
- Hyperbolic embedding replication (HyPoE / Poincaré, Marton 2025)
- Top-K NN Jaccard, item-level Procrustes displacement maps
- Gabor / corneal-V1 D_sens variants
- Per-subject reliability from the 50 within-subject repeated images

### Within-subject reliability

Each subject sees `n_double = 50` images in 2 different trials. We compute a per-subject
reliability index as the Spearman correlation of distances for repeated images' neighbours
across the two trials, and use it as an additional QC filter (lowest decile excluded).

### Subject de-duplication across SHINE cohorts

PID-hash assignment is injective: each Prolific ID maps to exactly one cohort, making
cross-enrollment structurally impossible within the single study. Every saved trial row
carries `shine_variant` (`"pre"` or `"post"`) for post-hoc auditing.

---

**Operational workflow** (implementation checklist, not part of OSF submission):
see [`WORKFLOW.md`](./WORKFLOW.md).
