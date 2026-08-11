# SpAM_Simulations

Simulate SpAM experiments (noisy subject distance judgements over a ground-truth embedding),
reconstruct them with weighted MDS, and measure how well two independent cohorts agree.

This file explains **what** the simulations answer, **which metrics** we use, and **how** the
pipeline is put together. For how to actually run things (quick start, R setup, EC2), see
[Cookbook.md](Cookbook.md). For what the runs have actually shown, including two results that were
later withdrawn, see [FINDINGS.md](FINDINGS.md).

## The current model is task-v5

**`task_v5_experiment` is the model this project uses. Everything in `sim_results/` predates it.**

v3 and v4 placed images on an *unbounded plane*. The deployed task places them in a rectangle and
divides every distance by its diagonal, so the observable cannot exceed 1.0 - but the unbounded
model produced a median per-trial maximum of **1.39**, i.e. arrangements that cannot physically
occur. [`models/canvas.py`](canvas.py) supplies the missing geometry and v5 makes it intrinsic.

| version | status |
|---|---|
| task-v0.1 / v2.3 / v2.4 | superseded, kept for their published sweeps |
| task-v3 | superseded; first generative coordinate-space model |
| task-v4, task-v4-fitted | **historical.** Every file under `sim_results/` came from these. Reproducible from current code (`test_bit_exact_v4`), which is why v4 is its own module rather than an edit - but not the current model |
| **task-v5** | **current.** v4's screening model on a bounded canvas. **Not yet run** |

Two consequences to carry into any reading of the older results. Their **comparative** findings
survive, since a geometry error shared by both arms of a contrast does not reverse it. Their
**absolute** figures do not - every asymptote and level was computed on an impossible distance
distribution. And their **calibrated constants do not transfer at all**, because
`subjects_noise_scale` changes meaning under v5 (an absolute fraction of canvas width, rather than a
ratio to each trial's arrangement spread), so stage 1 has to be re-run before a v5 sweep means
anything.

## Research questions

1. **Required N.** How many subjects does the study need, given a target metric to optimise? The
   answer depends on which metric you pick, which is why the metrics section below is long: a
   design that looks adequate under global rank agreement can be badly inadequate for recovering
   the closest pairs.
2. **Image-to-trial allocation: random vs engineered.** The deployed task shuffles images per
   participant and slices the result into trials, independently per subject and with no cohort-level
   coordination. A balanced incomplete block design instead chooses trials so every image *pair*
   co-occurs about equally often (MacDonald 2020, [10.3758/s13428-019-01326-x](https://doi.org/10.3758/s13428-019-01326-x)).
3. **Generalizability of the empirical data.** Given a real cohort, how far can its recovered
   structure be trusted: the geometry, the closest pairs, the cluster structure?

## Metrics

No single metric answers all three questions, and several of them disagree in informative ways.
Each entry says what it measures, and what it is good and bad at.

### Coverage and connectivity
| Metric | What it measures |
|---|---|
| `metrics.coverage` | % of images and of pairs observed at least once, mean observations per pair, number of connected components |

**Good at**: catching a design that cannot be analysed at all. Weighted MDS refuses a disconnected
pair graph, so `num_connected_components > 1` is fatal rather than merely bad.<br>
Image- and pair-coverage are also the only metrics that can be computed **before any subjects are recruited**,
so they are the only ones that can inform a *design choice* rather than a *sample-size choice*.
The deployed allocation is a random shuffle, which has statistically lower coverage than a balanced design,
as MacDonald 2020 shows, and our simulations confirm.<br>
**Bad at**: discriminating between workable designs. At the deployed session length the pair graph
is a single component for every arm at every N >= 30, so connectivity saturates and stops carrying
information. Pair coverage keeps discriminating well past that point.

### Cohort-to-cohort geometry
Every `rep` in a sweep is an **independently simulated subject-cohort**, so the rep-vs-rep comparison within
a configuration answers the study-planning question directly: *if I ran this study twice, would I get
the same answer?*

| Function | Compares                                               | Direction | Sensitive to |
|---|--------------------------------------------------------|---|---|
| `compute_embedding_stability` | reconstructed **distance vectors**, Spearman           | higher = better | rank order of the pairs |
| `compute_embedding_generalizability` | fitted **configurations** (=embeddings), Procrustes M² | **lower** = better | the recovered space itself, incl. metric distortion that leaves rank order intact |
| `compute_item_generalizability` | per-image residual after that alignment                | lower = better | *which* stimuli fail to generalise |

Procrustes centres, unit-norm scales and optimally rotates/reflects, exactly the gauge freedom an MDS
solution has, so what remains is genuine disagreement in relative geometry.

**Good at**: a single interpretable summary of whether the whole space reproduces. Procrustes catches
metric distortion that Spearman is blind to, and the two disagreeing is itself informative.<br>
**Bad at**: anything local. Both are computed over all 262,450 pairs, the overwhelming majority of
which are pairs of images from unrelated sub-categories, so they are dominated by the easy far-pairs.
A configuration can score 0.91 on global Spearman while its closest-pair structure is barely reproducible.

The configuration-space metrics need a store written with `run_mds_sweep(..., store_conf=True)` (the
default); older stores support only `compute_embedding_stability`.

### Closest-pair recovery
| Function | Compares | Direction |
|---|---|---|
| `compute_topk_similar_pair_stability` | the closest-`frac` **pair sets** of two cohorts, Jaccard | higher = better |
| `compute_recovery_vs_gt` | a cohort's closest-`frac` set against the **ground truth's**: recall, d-prime, plus threshold-free `separation_dprime` and `auc_near_pairs` | higher = better |

The first is reproducibility (do two cohorts agree), the second is validity (does a cohort find the
pairs that are genuinely closest). They can diverge, because two cohorts can agree on the wrong
answer, so both are reported. Recovery-vs-GT is simulation-only, since it needs a known truth.

**Good at**: the question the downstream stimulus construction actually poses, which is about near
neighbours and not about global geometry. At matched set sizes recall equals precision, so one
number carries it.<br>
**Bad at**: anything cluster-shaped. If A, B and C are three near-identical images, which of the
three pairs is "closest" flips with noise, and Jaccard scores every flip as an error even though all
three flips give the same practical answer (use one of them). It therefore *understates* usable
structure whenever the real question is about groups.

### Cluster structure
Discovered **bottom-up** from each cohort's embedding by agglomerative clustering, swept over
granularity `k` and over linkage (`average`, `ward`, `complete`). Linkage is the rule for the
distance between two *clusters*: average is the mean cross-cluster pair distance and assumes least;
ward merges whichever pair least increases within-cluster variance, which favours compact equal-sized
clusters; complete uses the farthest members. They can disagree on the same embedding, so all three
are swept.

| Metric | What it measures                                          | Direction                                     |
|---|-----------------------------------------------------------|-----------------------------------------------|
| **Variation of Information** | `H(A\| B) + H(B\|A)` between two cohorts' partitions      | **lower** = better |
| ARI, AMI | chance-corrected partition agreement                      | higher = better                               |
| cross-cohort silhouette, and **cross/within ratio** | labels from cohort A scored against cohort B's distances  | higher = better                               |
| cluster-wise max-Jaccard **distribution** | per cluster in A, its best match in B                     | higher = better                               |
| Baker's gamma | rank correlation of the two cohorts' cophenetic distances | higher = better                               |

**VI is primary, and specifically not ARI.** VI is a true metric on partitions (Meila 2003:
symmetric, non-negative, obeys the triangle inequality). That is what licenses chaining two claims,
`VI(cohort, reference) <= VI(cohort, cohort') + VI(cohort', reference)`. ARI is not a distance and
does **not** compose, so it must not be substituted there. Normalise VI by `log(n)` only, since a
constant divisor preserves metricity.

**Why the silhouette ratio and not raw silhouette.** Raw silhouette is a within-sample fit measure,
so it rewards overfitting; scoring cohort A's labels against cohort B's distances fixes that. It does
not fix the dimensionality confound, because distances concentrate at high D and depress silhouette
regardless of cluster quality. The `cross / within` ratio at matched (D, k) cancels that and the
small-k bias to first order, and reads as optimism: near 1 means the separation genuinely reproduces.

**Good at**: the granularity question. Reproducibility as a function of k tells you the finest
resolution the data supports, which is the level at which to deduplicate.<br>
**Bad at**: identifying the number of clusters, and telling you whether clusters exist at all. VI
scores *reproducibility*, and a coarse cut of a well-separated structure reproduces just as
perfectly as the right one: on three planted blobs VI is exactly 0 at both k=2 and k=3, because
every cohort merges the same two blobs. Cross-cohort silhouette is what distinguishes them, which is
why both `k_star_vi` and `k_star_sil` are reported. VI and ARI measure agreement on a partition,
not whether that partition is meaningful. Two cohorts can reproducibly agree on an arbitrary slicing
of a continuum, which is exactly why the silhouette ratio is carried alongside: high agreement with
near-zero cross-cohort silhouette is the signature of that failure, and it would mean "one image per
cluster" is the wrong rule and a distance threshold should be used instead.

`cluster_stability.continuum_diagnostics` reports that verdict explicitly per group, as `is_flat`
(the criterion varies by less than 0.02 across the whole k grid, so no granularity is
distinguishably better) and `is_arbitrary_slicing` (cross-cohort silhouette at k\* is below 0.05).
Both are **findings, not errors**.

Two implementation notes that affect how the numbers read. `fcluster(criterion="maxclust")` can
return **fewer** than k clusters when merge heights tie, so `n_clusters_realised` is recorded and no
consumer may assume it equals k. And `sil_ratio` is NaN wherever the within-cohort silhouette is not
meaningfully positive: a negative denominator makes the quotient flip sign, which on isotropic data
produced a ratio of +1.84 from a cross of -0.043, i.e. the metric appearing to claim that separation
*improved* out of sample. A negative ratio with a positive denominator is kept, since that
legitimately reports separation inverting across cohorts.

### Isolation: images that belong to no cluster

Agglomerative clustering assigns **every** image to a cluster, so an image that is genuinely
confusable with nothing gets absorbed into whichever group is nearest. At k=20 each cluster holds
~36 of the 725 images, so this is not a rare edge case, and a deduplication rule read off that
partition would exclude such an image for no reason. `density_clustering` runs HDBSCAN alongside the
agglomerative pass purely to recover the missing statement.

| Metric | What it measures | Direction |
|---|---|---|
| `frac_noise` | share of images HDBSCAN assigns to **no** cluster | descriptive |
| `noise_jaccard`, `noise_kappa` | do two cohorts agree on *which* images those are | higher = better |
| `vi_restricted` | VI over the images **all** labellings clustered | **lower** = better |
| `ari_shared_clustered` | ARI over the images both cohorts clustered | higher = better |
| `isolated_images.frac_cohorts_noise` | per image, the share of cohorts that left it unclustered | descriptive |

**The noise class blocks VI on the full set, but restricting the ground set recovers a real, if
narrower, chained claim.** `-1` is the absence of a cluster rather than a cluster, so counting it as
one would make VI dominated by a bucket that may hold most of the images. Drop the noise images from
*every* labelling involved, though, and what remains are honest partitions of one shared subset, on
which VI is a metric with all its usual properties. The claim becomes scoped rather than lost:
**restricted to the n\* images that all the labellings clustered**, the triangle inequality holds
exactly, so an unmeasured leg can still be bounded by the sum of two measured ones.

`common_clustered_mask` and `pairwise_restricted_vi` do this, and they take **all** the labellings at
once. Intersecting pairwise instead would score each pair on its own ground set, putting the terms in
different metric spaces where they cannot be added, and the bound would not follow.

The price is stated rather than hidden. The surviving subset is chosen *by the clusterings*, so two
cohorts that both label aggressively keep only the easy, well-separated core and score well on it:
`vi_restricted` is optimistically biased and the bias grows with the noise fraction. `n_shared` and
`frac_shared` therefore travel with every value, and `vi_restricted_norm` (divided by
`log(n_shared)`) is interpretable within a pair but **not** comparable across pairs with different
subset sizes. When chaining, add the raw nats.

**Good at**: the one question the agglomerative pass structurally cannot answer. An image left
unclustered by every cohort is one nothing is reliably confusable with, which makes it the safest
possible stimulus. HDBSCAN also *chooses* the number of clusters instead of being told, so it is an
independent read on granularity.<br>
**Bad at**: being a *primary* agreement measure. Its VI is conditional on a subset the method
itself chose, which is a weaker footing than the agglomerative pass's unconditional partition, and
the optimism above has no correction. It also swaps the `k` sweep for a `min_cluster_size` sweep
rather than removing a hyperparameter, and the noise fraction moves with that choice, so it is swept
(2, 3, 5, 10, 20) rather than reported at one setting.

### Clustering algorithms: why agglomerative, why HDBSCAN, why not GMM

The clusterer is a measurement instrument here, not a model of the stimuli, and the requirements
come from that. It must be **deterministic**, because any RNG inside it adds variance that is
indistinguishable from cohort disagreement and inflates the very quantity being measured. It must be
**rotation-invariant**, because two cohorts' MDS solutions differ by an arbitrary rotation and a
coordinate-dependent method would score identical geometries as disagreeing. And it must yield
**hard partitions** for the primary metric, because that is what makes VI a metric.

| | Agglomerative (primary) | HDBSCAN (descriptive) | GMM (rejected) |
|---|---|---|---|
| Deterministic | yes | yes | **no** (EM init) |
| Rotation-invariant | yes (runs on distances) | yes (`metric="precomputed"`) | only with full covariance |
| Can say "no cluster" | **no** | yes | no (soft, but every point has mass) |
| Partition for VI | yes | on the clustered subset only | no (soft) |
| Granularity control | `k`, swept | `min_cluster_size`, swept | `k`, swept |
| Whole k sweep from one fit | yes (one tree) | no | no |

**Agglomerative is primary** because it is the only one that satisfies all three requirements at
once, and because one linkage tree yields the entire k sweep plus Baker's gamma, the only k-free
metric in the set. Its cost is the hard-assignment blind spot, which is exactly what the HDBSCAN
pass above exists to cover.

**GMM is rejected on technical grounds, not preference.** At the granularities of interest there are
as few as 3.6 images per cluster (725 / 200), while a full covariance in 8-20 dimensions needs
36-210 parameters per component, so full covariance is unfittable. The fallback, diagonal or
spherical covariance, is **rotation-dependent** - and since cohort embeddings differ by an arbitrary
rotation, two cohorts recovering geometrically identical spaces would be scored as disagreeing. That
alone disqualifies it; EM's stochastic initialisation is a second, independent disqualification.
Gaussian components are also a density model, and MDS preserves distances, not densities.

**What this does not settle.** None of the above commits the *production* deduplication rule to a
clustering algorithm. If `is_flat` or `is_arbitrary_slicing` fires, the stated fallback is a plain
distance threshold, which needs no clusters at all. The clusterers are how the space is measured,
not how stimuli will ultimately be chosen.

### Empirical validity

| Function | What it measures |
|---|---|
| `validity.distribution_comparison` | simulated vs pilot distance distributions, median-rescaled (percentiles, CV, Wasserstein) |
| `validity.gradient_table` | mean distance by semantic level (same-leaf, same-subcategory, same-category, cross-category), as standardised gaps |
| `validity.noise_vs_distance` | RMSE between a pair's two judgements, binned by their mean distance: the shape of the **noise** rather than of the signal |

**Good at**: catching a simulation that is not realistic. The gradient was never fitted to, so it is
a genuine out-of-model check, and it is strong in the real pilot (same-leaf pairs 3.6x closer than
unrelated ones, a 1.53 SD gap), which means it has demonstrated power to detect a failure.<br>
**Bad at**: the distance-distribution half is nearly circular, since simulated subjects are generated
from a GT fitted to the pilot. A match there confirms little; a mismatch would still be a real alarm.
Distances also need median-rescaling first, because simulated distances are in GT-embedding units
while pilot distances are canvas-diagonal-normalised to [0, 1].

#### The noise-vs-distance curve, and why one half of it is expected to fail

Take every image pair a subject judged twice, plot the RMSE between the two judgements against their
mean, and the pilot gives an **inverted U**: clearly-similar and clearly-dissimilar pairs are judged
consistently, the ambiguous middle is where subjects disagree with themselves. It is a property of
the noise rather than of the signal, so a simulation can match the distance histogram perfectly and
still fail it.

**Measured on the real pilot** (47 subjects, 12,540 repeat pairs): the curve rises off a floor
(`rise_from_first` 0.83) to a **late** peak (`peak_bin_frac` 0.78) and then falls sharply
(`drop_from_peak` 0.37). The turnover is confined to the top bin, so `noise_curve_shape` describes
the curve **relative to its peak** rather than by comparing thirds. That is not a cosmetic choice:
an earlier thirds-based summary scored the pilot itself at `high_over_mid` 1.30 and therefore called
the real data *not* an inverted U, because averaging a high third smears the top-bin drop away.

The two ends are not equally strong tests. The low-end rise is close to forced, since distances
cannot go below zero and any additive-noise model reproduces it; a mismatch *there* would be a real
alarm. The **turnover** is the discriminating quantity, and it requires a bounded canvas: a pair
already at opposite corners cannot move much further apart, so its upper tail is truncated.

The check runs standalone from the calibrated parameters (`simulate_repeat_pairs`), since it
measures the noise model rather than any particular cohort size or allocation arm, and therefore
needs no MDS.

## How the pipeline works

| | Step                                                                     | Where |
|---|--------------------------------------------------------------------------|---|
| a | Calibrate to the pilot: fit the noise population, build the GT embedding | EC2 (needs R) |
| b | Generate noisy per-subject distances from the GT                         | EC2 |
| c | Pool `N` subjects into one cohort RDM, per allocation arm                | EC2 |
| d | Sweep `D`: fit weighted MDS to each cohort                               | EC2 (needs R) |
| e | Store configurations, distances and metadata                             | EC2 |
| f | Repeat b-e for `r` reps, each N, and each arm                            | EC2 |
| g | Sweep `k`×`linkage` over the stored embeddings to get partitions         | local |
| h | Compute between-cohort metrics over the $C(r,2)$ rep pairs               | local |
| i | Select `k*`, `D*`, required-`N`; run the continuum diagnostics               | local |

Steps g-i are deliberately **post-processing**. Agglomerative clustering on 725 points is
milliseconds against ~25 seconds for one SMACOF fit, so keeping it out of the expensive loop costs
nothing and lets k, linkage and metric choices be revised without refitting any MDS.

### Parameter ledger

Every free parameter, what identifies it, and **which subjects it was measured on**. The cohort
column is not bookkeeping: production data must never calibrate a simulation whose purpose is to
plan the production study, and the bases genuinely differ because different observables need
different data.

| parameter | identified by | cohort basis | status |
|---|---|---|---|
| `n_dims` | split-half agreement + leave-k-out CV | 41 pilot, **pre-SHINE only** | selected out-of-sample |
| `subjects_noise_scale` | within-subject test-retest | 47 pilot, both variants | calibrated |
| noise family / shape | per-subject reliability distribution | 47 pilot, both variants | calibrated |
| `perspective_dispersion` | between-subject agreement | 47 pilot, both variants | calibrated |
| `canvas.aspect` | screen geometry | 26 pilot, **v3+ only** | measured, or sampled |
| `canvas.fill` | per-trial max distance | 26 pilot, **v3+ only** | calibrated, or sampled |
| `canvas.softness` | near-wall placement density | 26 pilot, **v3+ only** | **swept** (`canvas_softness`, sensitivity) |

**Why the three bases differ.** The ground truth is a geometry over a *stimulus set*, so it takes
pre-SHINE subjects only: the post-SHINE half judged different images and one geometry cannot
describe both. The noise fit estimates a property of *subjects* - how reliably a person repeats
their own arrangement - so the post-SHINE half is admissible evidence and all 47 are used. The
canvas constants need trial-level `final_locations`, which pre-v3 sessions do not record, leaving
26. **That last base is thin and should be reported as such.**

An earlier version of the canvas constants was measured over 114 v3+ participants with no cohort
filter, 88 of them production. That is the same contamination recorded for the task-v4 run in
[FINDINGS.md](FINDINGS.md), caught and corrected here. `aspect` turned out to be cohort-invariant
(0.494 against 0.499), which is what one expects of screen size, but the pilot value is used anyway
rather than arguing an exception.

**Sampled rather than fixed.** `aspect` and `fill` are *observable* and have measured
distributions, so they are resampled per subject/trial from the empirical quantiles
(`canvas.sample_spec`) rather than pinned at a median. That costs no free parameter and fixes a real
mismatch: with a fixed spec the simulated per-trial max distance has sd 0.039 against the pilot's
0.106, and sampling recovers 0.089 of it. `softness` has no observable distribution - nothing in the
data yields the exponent directly - which is exactly why it is swept as a sensitivity axis instead.

**What this buys against the obvious objection.** A model with seven dials fitted until it matches
is not evidence of anything. The defence is that each parameter is anchored to a *distinct*
observable with a stated identification argument (test-retest and between-subject agreement
identify noise and dispersion *sequentially*, not jointly), that two of them are resampled rather
than fitted, that one is swept rather than chosen, and that the semantic gradient is a genuine
out-of-sample check nothing was fitted to. The simulation is a noise-and-sampling model for power
analysis, not a cognitive theory of SpAM, and the conclusions are reported with sensitivity to the
parameters that could plausibly drive them.

### Step a: calibrate to the pilot

Without calibration the simulation's internals are guessed, so the absolute required-N is only as
meaningful as the guesses. `empirical/pilot.py` anchors them to real data, turning the estimate from "as a
function of guessed internals" into a calibrated number.

Three observables do the work. **Ground-truth geometry** comes from pooling the pre-SHINE pilot
subjects into one aggregate RDM and running weighted SMACOF; the recovered embedding *is* the GT, and
it inherits the real cluster structure. **`subjects_noise_scale`** is pinned by within-subject
test-retest from the v3.x whole-trial repeats, which is perspective-invariant: a repeat re-projects
to the same 2-D arrangement and differs only by fresh placement noise, so it isolates placement
precision. **`perspective_dispersion`** is then pinned by between-subject agreement with noise held
fixed. Because the model makes the two levers triangular (test-retest = f(noise), agreement =
g(noise, dispersion)), identifiability is sequential and exact.

> **Cohort policy.** Simulations calibrate on the **pilot cohort only**. Calibrating on production
> data would shape the sample-size and screening conclusions using the very cohort they are meant to
> plan. Note every v4.0 session is `production`, so the pilot view carries **no screening-block
> data**; reliability comes from v3.x whole-trial repeats.

> **SHINE-variant policy. Cohort is not a proxy for variant.** `task.js` documents pilot sessions as
> always pre-SHINE and `docs/WORKFLOW.md` repeats it, but the data disagrees: of the 47 loadable
> pilot subjects, **41 are `pre` and 6 are `post`** (all v3.06). Because `pilot._src_to_relpath`
> strips the `<variant>_shine` path segment, both variants map onto the same manifest index and
> pooling them is silent. **Ground-truth construction must pass `variants=("pre",)`**, since two
> variants are two stimulus sets and one geometry cannot describe both. **Noise-model fitting is
> exempt** and may use all 47: it estimates a property of subjects, not of the images.

> **Data policy.** `data/` is human-subjects data: gitignored, **never committed or pushed**.
> Pilot-derived artifacts (the aggregate RDM, the GT `coords`, fitted params) are equally local: keep
> them under the private `$S3_URI` prefix, never in the repo.

#### Choosing the GT dimensionality

`n_dims` used to default to a rule that read a classical-MDS eigenspectrum off the **mean-imputed**
aggregate RDM and took the smallest dimensionality explaining 90% of variance, capped at 15. That
rule is invalid here and has been **removed**; `build_gt_from_pilot` now requires `n_dims`.

63.6% of pairs are unobserved and were filled with a single constant. That asserts every one of those
pairs is equidistant, and *k* mutually equidistant points form a regular simplex requiring *k*-1
dimensions, so the fill manufactures rank rather than merely adding noise. Measured: a synthetic
**rank-8** space put through the identical mask and fill reports an effective rank of **193** and
needs **239** dimensions for 90% variance, statistically indistinguishable from the real data's 213
and 216. The rule therefore returned its cap, carrying essentially no information, and the resulting
GT was near-isotropic (4.9% to 9.5% variance per dimension across 15).

It is replaced by out-of-sample prediction, with no imputation anywhere (weighted SMACOF treats
weight 0 as missing): `gt_construction.dimensionality_scan` fits each candidate dimensionality on
disjoint halves of the subjects and scores agreement, and `cross_validate_ndim` verifies by
leave-k-out over subjects. Three details matter. The **same splits are reused across every `ndim`**,
making the comparison paired, which is load-bearing because the curve is expected to be nearly flat.
**Discarding disconnected halves is a biased filter**: at ~20 subjects a half is connected only ~90%
of the time, and it fails precisely when it holds poorly-covered subjects, so `draw_valid_splits`
reports the discard rate and the coverage gap between kept and discarded draws. And `select_ndim`
defaults to the **one-SE rule**, the smallest `ndim` within one standard error of the best, because
on a flat curve a plain argmax is noise-driven and drifts high.

`method="classical"` exists for running this without R. It mean-imputes, i.e. does the very thing
described above, so it is a **plumbing smoke test only** and must never select a dimensionality.

This whole step is its own EC2 stage, `ec2/run_gt_construction.sh`, and its own instance. The scan
is 1100 fits and the CV another 440, so `gt_construction` carries a joblib payload path alongside
the readable serial one (`split_aggregates`, `scan_ndim_parallel`, `cross_validate_ndim_parallel`),
mirroring `pipeline`'s: subjects are pooled into arrays once, R is imported inside each worker, and
a failed fit is recorded rather than raised so one bad draw cannot abort a multi-hour scan. The
drivers take **one `ndim` per call** so the loop checkpoints to S3 after each dimensionality. The
stage writes `gt/selection.json`, which is what supplies `N_DIMS` to every later script.

### Steps b-c: generate cohorts

Each simulated subject gets a placement precision drawn from the fitted noise population, a
perspective weighting of the GT's dimensions, and a set of trials. Per trial, the weighted GT
coordinates are projected to a local 2-D arrangement (the SpAM canvas bottleneck) and perturbed by
placement noise. Under task-v4 each candidate first completes a screening block and is retained only
if their minimum per-repeat test-retest clears `screening_min_reliability`.

Which images land in which trial is the **allocation arm**, swept as the numeric `allocation_mode`
lever: `0.0` reproduces the deployed per-subject shuffle, `1.0` draws pre-generated balanced sessions
from `block_design.greedy_session_design`. A screened-out candidate returns its session to the pool,
so the design remains a plan over *analysed* subjects rather than degrading in proportion to the
rejection rate.

### Steps d-f: fit and store

`run_mds_sweep` fits weighted SMACOF at every requested dimensionality for every (config, rep),
streaming each result into a `ResultStore`. The store keeps a human-readable `meta.csv` plus flat
float32 binaries for the reconstructed distances and the fitted configurations. It is resumable:
re-running skips work already recorded, which is the only checkpointing an interrupted EC2 sweep has.

### Steps g-i: cluster and decide

These run **locally**, on a downloaded store, and need no R. `cli/run_cluster_analysis.py` is the
one-command driver:

```bash
python -m SpAM_Simulations.cli.run_cluster_analysis --store <run>/mds_store --out <run>/out
```

It writes six frames that `eval_helpers.load_run` picks up as optionals: `cluster_agreement.csv`,
`dendrogram_agreement.csv`, `cluster_sizes.csv` and `k_selection.csv` from the agglomerative pass,
plus `density_agreement.csv` and `isolated_images.csv` from the HDBSCAN one. Or call the pieces
directly:

```python
from SpAM_Simulations.measures import cluster_stability as cs
from SpAM_Simulations.measures import density_clustering as dc
from SpAM_Simulations.core.storage import ResultStore

store = ResultStore.open("sim_results/<run>/mds_store/<cell>")
agreement = cs.compute_cluster_agreement(store)      # per (config, ndim, linkage, k)
sizes     = cs.compute_cluster_sizes(store)
trees     = cs.compute_dendrogram_agreement(store)   # k-free
density   = dc.compute_density_agreement(store)      # per (config, ndim, min_cluster_size)
isolated  = dc.isolated_images(store)                # per image: how often unclustered
verdicts  = cs.continuum_diagnostics(agreement)      # k*, is_flat, is_arbitrary_slicing
```

Each cohort is clustered at every (k, linkage) and compared against every other rep in its group, so
the C(r,2) comparisons answer "would a second run recover the same clusters?". Distances are
recomputed from the stored **coordinates** rather than read from `confdists.f32`: the two are equal,
but coordinates are ~20x smaller (28 KB against 1.05 MB per fit, measured), so only `confs.f32` need
be downloaded. Recomputing also guarantees the vector is Euclidean in `ndim` dimensions, which is
what makes Ward linkage well-defined.

Each fit is prepared once per group (`pdist`, `squareform`, three linkages, all cuts, cophenetic
ranks). Linkage and cutting are O(reps) while the comparisons are O(reps²), so doing them inside the
pair loop would rebuild every tree r-1 times. Measured at 725 images: **1.1 s per rep pair** across
the full 12-k × 3-linkage grid, so a 2,160-pair sweep is ~40 min single-core.

`select_k` then applies the same **one-SE rule** used for dimensionality, via the shared
`gt_construction.apply_selection_rule`, taking the smallest k within one standard error of the best.
For a deduplication rule the parsimonious end is the safe one: a coarser k merges more images and
excludes more candidate pairs, which is the conservative error.

**Two granularities are reported, because VI does not identify the number of clusters.** VI measures
*reproducibility*, and a coarse cut of a well-separated structure reproduces perfectly too. On three
planted blobs, VI is exactly 0 at k=2 and at k=3 alike (every cohort merges the same two blobs), so
the parsimony tiebreak returns 2; cross-cohort silhouette is what tells them apart, peaking at the
true 3 (0.93 against 0.76). So `k_selection.csv` carries both: `k_star_vi`, selected on VI, is the
coarsest granularity that reproduces and is the safe deduplication rule; `k_star_sil`, selected on
cross-cohort silhouette, is where the structure actually is. Reading k\* off VI alone would
systematically under-report the granularity the data supports.

All three headline metrics (`vi_norm`, `sil_cross`, `sil_ratio`) are recorded at **both** k\*, as
`<metric>_at_k_star_vi` and `<metric>_at_k_star_sil`. That is what makes the trade readable in
either direction: how much separation the parsimonious choice gives up, and how much
reproducibility the structured one costs. On the planted blobs the finer `k_star_sil` turns out to
be **free** - VI is 0 at both - which is a conclusion you cannot reach from the VI-side columns
alone, since those only show that the two k\* differ.

The density pass runs over the same prepared distances and is reported separately in both the CSVs
and the printed summary. Keeping the two apart is deliberate: its noise-class scores do not compose,
so letting them sit in the same frame as VI would invite exactly the substitution the
[clustering-algorithms section](#clustering-algorithms-why-agglomerative-why-hdbscan-why-not-gmm)
rules out.

> One caveat inherited from every rep-pair metric here: the C(r,2) pairs are **not independent**,
> since each cohort appears in r-1 of them, so the reported SEM understates the true uncertainty.

## Modules

```
SpAM_Simulations/
  models/     the generative task models        core/       orchestration, storage, MDS
  measures/   everything that measures          empirical/  pilot, ground truth, calibration
  reporting/  report + notebook helpers         cli/        entry points
  notebooks/  *.ipynb                           ec2/        provisioning + sweep entrypoints
  __tests__/  the suite                         sim_results/  downloaded runs (gitignored)
```

There is no `__init__.py` anywhere: the repo root is the package root and these are namespace
packages, so run everything from the repo root.

### `models/`

The generative task models: how a simulated participant produces an arrangement.

| Module | Responsibility |
|---|---|
| `experiment.py` | Core simulation: `simulate_experiment` / `simulate_single_subject` (vectorized, condensed form). |
| `design.py` | Per-subject trial allocation (`compute_design_counts`, `build_trial_lists`, `distinct_trial_count`, `select_repeat_trials`), ported from `SpAM_Task`'s `buildTrialLists`/`insertTrialRepeats`. |
| `task_v2_3_experiment.py` | Task-v2.3: per-subject image subset + trial design, plus the within-subject SNR heuristic. |
| `task_v2_4_experiment.py` | Task-v2.4: v2.3 design **plus** `frac_trials_repeated` whole-trial repeats, yielding per-subject test-retest. Bit-exact to v2.3 at `frac_trials_repeated=0`. |
| `task_v3_experiment.py` | Task-v3: a **generative coordinate-space** model. Per subject, a perspective weighting of the GT dimensions projected onto a local per-trial 2-D arrangement, then canvas placement noise. |
| `task_v5_experiment.py` | **Task-v5: the v4 model on a bounded canvas.** Thin wrapper injecting `canvas.make_canvas_trial_simulator` through v4's `trial_simulator` seam, so nothing is duplicated and v4 stays bit-exact. The canvas is intrinsic, not a flag. |
| `canvas.py` | The 2-D sort rectangle: `CanvasSpec`, per-axis `fit_to_canvas`, smooth `soft_bound` walls, `sample_spec` for per-trial screen shapes. Constants measured on the 26 pilot-cohort v3+ subjects. |
| `task_v4_experiment.py` | Task-v4: the v3 model **plus the deployed v4.0 screening block**, and the `allocation_mode` arm. Candidates are recruited until `num_subjects` pass, capped by `MAX_RECRUIT_PER_SUBJECT`. `screening_trials=0` reduces to v3 bit-for-bit. |
| `block_design.py` | Balanced incomplete block designs (MacDonald's "best of greedy", vectorised): `greedy_design`, `best_of_greedy`, `schonheim`, `greedy_session_design`. |
| `allocation.py` | The `allocation_mode` arm: `RandomAllocator` (deployed scheme) and `DesignedAllocator` (balanced sessions, with rollback). |

### `core/`

Orchestration and persistence: configure a sweep, run it, store it.

| Module | Responsibility |
|---|---|
| `simulation.py` | `Simulation` container + ground-truth distances; `make` / `from_embeddings` / `build_ground_truth_embeddings`. |
| `helpers.py` | Distance-matrix format conversion (`convert_to_condensed`). |
| `multi_dimensional_scaling.py` | `run_mds`: weighted SMACOF via R's `smacof` (needs R + rpy2). |
| `config.py` | `SimulationConfig` and the task-v2.3 / v2.4 / v3 / v4 / v5 variants (v4 carries the `allocation_mode` arm, v5 adds `canvas_softness`), plus `MDSSweepConfig`. |
| `pipeline.py` | Orchestration: generate (v0.1 through v5), coverage, stability, MDS sweep, embedding/item generalizability, top-k stability, `compute_recovery_vs_gt`. |
| `storage.py` | `ResultStore`: compact, streamable, resumable on-disk store. Holds each fit's `confdist` and, optionally, its MDS configuration (`confs.f32`). Format v2; v1 stores still open unchanged. |

### `measures/`

Everything that measures a cohort or an embedding. No simulation, no R.

| Module | Responsibility |
|---|---|
| `metrics.py` | `coverage`, `spearman_correlation`, `snr_summary`, `test_retest_summary`, `screening_summary`, `effective_rank`, `topk_similar_jaccard`. |
| `design_comparison.py` | Compares allocation arms as **sampling plans** (coverage, per-image balance, waste). No subjects, no MDS, no R. |
| `recovery.py` | Recovery of the GT's closest pairs: `recall_at_frac`, `dprime_at_frac`, `separation_dprime`, `auc_near_pairs`. |
| `validity.py` | Is a simulated cohort realistic: distance-distribution comparison, the semantic-hierarchy gradient, and the noise-vs-distance curve (`noise_vs_distance`, `noise_curve_shape`, `simulate_repeat_pairs`). |
| `density_clustering.py` | **Descriptive** density pass (HDBSCAN): noise fraction, cross-cohort agreement on *which* images are isolated, `compute_density_agreement`, `isolated_images`. Never enters the VI chain. |
| `cluster_stability.py` | Between-cohort cluster agreement: VI/ARI/AMI, cross-cohort silhouette, cluster-wise Jaccard, Baker's gamma; the `compute_cluster_*` store drivers; and `select_k` / `continuum_diagnostics`. Runs **locally** on a downloaded store. |

### `empirical/`

The pilot data, the ground truth built from it, and the calibration that ties the model to it.

| Module | Responsibility |
|---|---|
| `pilot.py` | **Read-only** pilot ingestion + calibration: load the flat `data/` dir, filter by cohort / version / **SHINE variant**, the test-retest and between-subject-agreement observables, and `calibrate_params_from_pilot`. |
| `gt_construction.py` | **Task-agnostic** GT construction: `dimensionality_scan`, `select_ndim` (one-SE), `cross_validate_ndim`, `build_gt`, plus the joblib payload path (`split_aggregates`, `scan_ndim_parallel`, `cross_validate_ndim_parallel`) the EC2 stage needs. Replaces the retired imputed-eigenspectrum rule. |
| `calibrate_v5.py` | The three fitted constants (noise population, dispersion, test-retest noise) with a **fingerprint-checked cache**. Lifted out of the EC2 heredoc so it is testable and re-runnable locally. The fingerprint hashes the GT *coordinates*, not its filename. |
| `gt_diagnostics.py` | Is a fitted GT worth trusting: per-level observed coverage, its own semantic gradient, its in-sample agreement with the raw aggregate, and the **half-split noise ceiling** that makes that agreement interpretable. Cheap screen, no R, no MDS. |

### `reporting/`

Turning a finished run into something readable.

| Module | Responsibility |
|---|---|
| `build_report.py`, `report_sections.py`, `report_clusters.py` | Build the self-contained HTML results report. Every number is read from the run's CSVs at build time, so the page cannot drift from the tables. `build_report` also exposes a CLI: `python -m SpAM_Simulations.reporting.build_report`. |
| `eval_helpers.py` | Read-only loading/plotting helpers for `notebooks/evaluate_simulation.ipynb`. No simulation, no MDS, no R. |

### `cli/`

Entry points. `python -m SpAM_Simulations.cli.<name> --help` for each.

| Module | Responsibility |
|---|---|
| `build_extra_gt.py` | Builds a GT at a chosen dimensionality, for when the scan's choice is a sample-size floor rather than the intrinsic dimensionality. Records the build without touching `selection.json`. |
| `run_cluster_analysis.py` | Local CLI driver for pipeline steps g-i: opens a downloaded store, writes the six cluster tables, prints the continuum verdicts. No R, no EC2. |
| `run_validity.py` | Local CLI for the noise-vs-distance check, driven from a downloaded run's `calibration.json`. Needs no store and no R; runs in seconds. |
| `recompute_store_tables.py` | Rebuilds the four store-derived metric tables locally, for when a grouping or de-duplication fix lands after the EC2 run. |
| `example_pipeline.py` | Minimal runnable end-to-end example. |

### `notebooks/` and other paths

| Path | Responsibility |
|---|---|
| `notebooks/report_v5.ipynb` | Notebook companion to the report: same tables, same figures, for checking or re-cutting any of it. |
| `notebooks/evaluate_simulation.ipynb` | Overview/drill-down figures for a completed run, via `reporting/eval_helpers.py`. |
| `notebooks/evaluation_v0_1.ipynb`, `notebooks/evaluation_task_v2_3.ipynb`, `notebooks/evaluation_task_v2_4.ipynb` | Per-task-version plotting notebooks for the older simulations. |
| `ec2/` | Provisioning + staging helpers (`prepare_machine.sh`) and the sweep entrypoints, including the current two-stage programme (`run_gt_construction.sh`, `run_design_comparison.sh`). See [Cookbook.md](Cookbook.md#running-on-ec2). |
| `sim_results/<run-name>/` | Local copy of a completed run's small files, downloaded from S3. Gitignored. |

## Running the cluster analysis at scale

The post-MDS cluster metrics walk every configuration group in a store, and a full stage-2 store is
1,728 groups of 10 fits over 725 images. Three things make that tractable, all on by default:

* **Groups are the parallel axis** (`n_jobs=-1`). They are independent, and workers reopen the store
  from its path rather than receiving it, because a `ResultStore` holds file handles and memmaps that
  do not survive pickling. `n_jobs=1` runs in-process and is what the tests use.
* **Rep pairs are sampled, not exhausted** (`max_pairs=22` of `C(10,2)=45`). Roughly a 40% cut in wall
  clock for a ~1.4x wider SEM, on an estimate whose pairs were never independent anyway. Pass
  `--max-pairs 0` to exhaust them.
* **One traversal, three tables.** `compute_agglomerative_tables` returns agreement, dendrogram and
  sizes from a single pass; computed separately they each rebuilt the same linkage trees, cuts and
  cophenetic rankings.

Even so this is hours, not minutes, and it is memory-bandwidth bound rather than compute bound: the
dominant cost is `silhouette_score` sweeping precomputed 725x725 matrices, and eight concurrent
workers do not have eight times the cache.

**De-duplication.** `pipeline._grouped_successful` keeps one row per `(configuration, rep)`. An
append-only store that is resumed without recognising completed work holds two copies of every fit it
redoes, and a duplicate enters the pair loop as a self-comparison of an identical cohort - VI exactly
0, ARI exactly 1 - biasing every rep-pair metric upward.

## Tests

```
.venv/Scripts/python.exe -m pytest SpAM_Simulations/__tests__ -q
```

549 tests, 5 skipped without R. R-dependent tests (`test_pipeline_mds.py`, one case in
`test_gt_construction.py`) auto-skip if the bridge can't be imported; the rest run anywhere.

Two suites pin behaviour that must not drift. `test_bit_exact.py` checks `simulate_experiment`
against a fixture recorded before the vectorization refactor, and `test_bit_exact_v4.py` does the
same for the task-v4 model, recorded before the `allocation_mode` arm was added. A failure in either
means the RNG stream moved and previously published runs are no longer reproducible.

## Tags

Named snapshots at notable `SpAM_Simulations` milestones:

| Tag | Commit | Date | Marks |
|---|---|---|---|
| `spam-sim-pre-refactor` | `73a9b14` | 2026-06-18 | Last commit before the bit-exact vectorization + reusable pipeline/storage refactor (condensed-form simulation, ~9x speed-up; `ResultStore`; parallel MDS sweep; EC2 provisioning). |
| `sim-v2.3` | `33d65e2` | 2026-06-24 | `notebooks/evaluate_simulation.ipynb` display/correctness polish plus the `ec2/`/`sim_results/` directory reorg. |
| `spam-task-v2.5` | `b82f454` | 2026-06-25 | SpAM task v2.5. |
| `spam-sim-v3-calibrated` | `813896d` | 2026-07-07 | GT-build provenance and the first pilot-calibrated task-v3 sweep. |
| `spam-task-v3.24` | `7cec31b` | 2026-07-08 | SpAM task v3.24. |
| `spam-sim-v4.0` | `53d667c` | 2026-07-26 | Noise population fitted to data. **The revision every file under `sim_results/` was produced from** - historical, not current - and the last one carrying the v1 parser and the pilot/prod directory layout. |
| *(untagged)* | — | 2026-08 | **task-v5**: the bounded canvas, cluster-reproducibility metrics, the density pass, and the two-stage EC2 programme. Tag when stage 1 has been re-run under v5. |
