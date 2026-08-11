# SpAM simulation findings

What the simulation runs have established, what they have not, and which conclusions were later
withdrawn. Every number here is read from a committed result file or recomputed from `data/`; the
source is named so each can be re-derived.

For *how* the pipeline works see [README.md](README.md), and for how to run it
[Cookbook.md](Cookbook.md).

> ## ⚠️ Every run described here predates task-v5, and none has been re-run under it
>
> **The model these runs used placed images on an unbounded plane.** The deployed task places them
> in a rectangle and divides every distance by its diagonal, so the observable cannot exceed 1.0 -
> yet the v3/v4 model produced a median per-trial maximum of **1.39**. It was generating
> arrangements that cannot physically occur. `canvas.py` fixes this and `task_v5_experiment` is the
> current model; see [README.md](README.md#the-current-model-is-task-v5).
>
> What that means for the numbers below:
>
> * **Comparative findings survive.** Screening's direction and cost, "reliability beats sample
>   size", reproducibility ranking designs the same way validity does - these are contrasts between
>   arms measured under one consistent model, and a geometry error shared by both arms does not
>   reverse them.
> * **Absolute figures are provisional.** Every asymptote, every Spearman/M² level, every recall and
>   Jaccard number was computed on distances drawn from an impossible distribution. Treat them as
>   ordinal, not as quantities to plan against.
> * **Calibrated constants do not transfer at all.** `subjects_noise_scale` changes meaning under
>   v5 (absolute canvas fraction, not a ratio to arrangement spread), so the fitted values here
>   cannot be reused; stage 1 must be re-run before any v5 sweep means anything.
>
> Three further caveats, flagged inline rather than left for a reader to notice: two results were
> revised after the fact, the headline required-N figure is a **non-result** (the sweep never
> converged), and the v4-fitted calibration set included 14 production subjects alongside the pilot.

---

## 1. The empirical substrate

Everything downstream is anchored to the pilot, so its properties bound what any simulation can
claim.

| Fact | Value | Source |
|---|---|---|
| Pilot subjects loaded | 47 of 51 sessions | `pilot.load_pilot_subjects` |
| Pre-SHINE / post-SHINE | **41 / 6** (all post are v3.06) | `participants.shine_variant` |
| Pair coverage, pre-SHINE pooled | **31.5%** of 262,450 pairs | `gt_construction.aggregate_subjects` |
| Pair coverage, all 47 | 36.4% | as above |
| Observations per observed pair | mean 1.37; **70.8% seen by one subject** | recomputed from `data/` |
| Subjects per image | median 16 (range 7 to 26) | recomputed from `data/` |
| Per-subject test-retest | median 0.243 over 47; 22 have retest data | `pilot.within_subject_test_retest` |

Three consequences, each of which shaped a later design decision.

**Cohort is not a proxy for SHINE variant.** `task.js` assigns pilot sessions the pre-SHINE images
and the docs repeat it as fact, but 6 pilot sessions recorded `post`. Since `pilot._src_to_relpath`
strips the variant segment, both variants mapped onto the same manifest index and were pooled
silently into every GT built before this was found.

**The cohort-level RDM has low per-pair reliability, and adding subjects does not fix it.**
Split-half Spearman is **0.15** (Spearman-Brown corrected to **0.26** for the full cohort), and it is
flat across cohort size:

| cohort N | 8 | 16 | 24 | 32 | 46 |
|---|---|---|---|---|---|
| RDM reliability | 0.250 | 0.269 | 0.254 | 0.250 | 0.256 |

The reason is that coverage **broadens rather than deepens**. From N=5 to N=47, pairs covered rises
4.8% to 36.4% while observations per covered pair moves only 1.13 to 1.37. Each new subject mostly
discovers pairs nobody has judged rather than replicating existing ones, because 47 subjects produce
131,100 pair-observations over a 262,450-pair space.

**But the aggregate carries strong semantic structure.** Pooled pre-SHINE distances by hierarchy
level (`validity.gradient_table`):

| level | n pairs | mean distance | standardised gap |
|---|---|---|---|
| unrelated | 39,779 | 0.374 | 0.00 |
| cross-category | 20,441 | 0.354 | 0.11 |
| same-category | 15,835 | 0.293 | 0.46 |
| same-subcategory | 5,014 | 0.184 | 1.08 |
| same-leaf | 1,476 | **0.104** | **1.53** |

Cleanly monotone, with same-leaf pairs 3.6x closer than unrelated ones. Low per-pair reliability and
strong aggregate structure are compatible: individual estimates are noisy, but averaged over
thousands of pairs per level the signal is unmistakable. **This matters for interpretation.** The
gradient is a between-level contrast with thousands of pairs each side, where noise averages away.
The stimulus-construction question is a within-level discrimination among near-ties, which is
exactly where per-pair noise bites.

---

## 2. Run history

| Run | Model | What it added |
|---|---|---|
| task-v0.1 | additive distance noise | first convergence sweep |
| task-v2.3 | per-subject trial design matching the deployed task | realistic sampling |
| task-v2.4 | + whole-trial repeats | per-subject test-retest as a lever |
| task-v3 | **generative coordinate-space model** | perspective weighting + canvas placement noise; first pilot-calibrated GT |
| task-v4 | + the deployed v4.0 screening block | recruitment cost, retained-cohort precision |
| task-v4-fitted | + noise *shape* fitted, not just scale | the last run on the **unbounded** model |
| **task-v5** | + the **bounded 2-D canvas** | distances that can physically occur; **not yet run** |

Tag `spam-sim-v4.0` (`53d667c`) is the revision every result file under `sim_results/` was produced
from. It is a **historical** revision, not the current one: `test_bit_exact_v4` guarantees the
current code can still reproduce it, which is exactly why v4 was kept as its own module rather than
edited in place, but no conclusion below should be read as describing the model the project now
uses.

---

## 3. Findings carried forward from task-v3

Sweep: N ∈ {30, 50, 75, 300}, 5 reps (so C(5,2) = 10 cohort pairs per cell), designs of 20 or 25
trials/images, `perspective_dispersion` 0.2.

### Recovery of the ground truth's closest pairs is poor

`sim_results/task-v3/recovery_vs_gt.csv`, at ndim=8 and the 20-trial/20-image design, top 5%:

| N | recall @ R=0.24 | @ R=0.35 | @ R=0.50 | @ R=0.65 |
|---|---|---|---|---|
| 30 | 0.102 | 0.145 | 0.217 | 0.285 |
| 75 | 0.152 | 0.199 | 0.266 | 0.338 |
| 300 | **0.217** | 0.265 | 0.353 | 0.451 |

At the pilot-realistic reliability (R≈0.24) and N=300, a cohort recovers **22% of the truly-closest
pairs**. d-prime at that cell is 0.95.

**Reliability dominates sample size.** N=30 at R=0.65 (recall 0.285) beats N=300 at R=0.24 (0.217).
Going from N=30 to N=300, a tenfold increase, moves recall 0.102 to 0.217. This is the single most
actionable result in the programme: per-subject data quality buys more than recruitment does.

### Reproducibility and validity rank designs identically but differ in level

Joining `topk_jaccard.csv` and `recovery_vs_gt.csv` on 3,120 matched cells:

- Spearman rho between cohort-vs-cohort agreement and cohort-vs-GT recovery: **0.98** (Pearson 0.94).
- But levels differ by **2 to 4x**, with cohort-vs-cohort always the pessimistic one.

So for *choosing* a design they are interchangeable, and cohort-vs-cohort is preferable since it
assumes no ground truth. For *absolute* claims they disagree substantially.

---

## 4. task-v4-fitted: the last unbounded run

Design fixed to the deployed `task_config.json`: 20 images/trial, 8 screening trials (2 repeats) plus
14 experimental (2 repeats). Sweep: N ∈ {30, 50, 75, 300} x ndim {5,6,8,10} x
`perspective_dispersion` {0.15, 0.3, 0.45} x `screening_min_reliability` {-1, 0, 0.2, 0.4} x target
test-retest {0.24, 0.35, 0.50}, **6 reps** (C(6,2) = 15 cohort pairs per cell).

### Calibration

Fitted against the pilot, in `calibration/`:

```
noise family      lognormal, shape (sigma) = 0.35
                  sim median 0.225 vs empirical 0.271 (n=36), CV 0.361, not at the shape boundary
perspective_dispersion   fitted 0.30   (empirical agreement 0.145, achieved 0.141)
noise_map:  target TR 0.24 -> noise_scale 1.5 -> achieved unscreened 0.244
            target TR 0.35 -> 1.2 -> 0.344
            target TR 0.50 -> 0.9 -> 0.487
```

#### The calibration set included 14 production subjects

The run's `calibrate.log` reports `loaded 61 completed sessions` at 46.5% pair coverage, with a
reliability sample of n=36. The pilot cohort is 47 subjects at 36.4% coverage, 22 of them with
retest data. Adding 14 v4.\* production subjects reproduces all three numbers at once (61 sessions,
46.6% coverage, 36 with retest), and no other count fits.

The cause is that the loader at tag `spam-sim-v4.0` defined "pilot" by **folder membership**
(`PILOT_DIR="data/pilot"`, no cohort or version filter), so whatever was staged into that S3 prefix
became the calibration set.

What this shifts, in rough order of concern:

- The **noise-shape fit** leans on those 14 more than their share of sessions suggests, since every
  v4 subject has screening repeats and therefore contributes retest data, making them 39% of the
  reliability sample. They had also already passed v4.0's deployed screening, so the distribution is
  post-screening while the simulation applies it to unscreened candidates. That biases the noise
  model slightly optimistic.
- The **GT geometry** was built on all 61, mixing both cohorts and both SHINE variants.
- It explains the **two test-retest medians**: 0.271 (n=36, this run) against 0.243 (47 pilot
  subjects, recomputed cleanly). The higher value is the screened v4 subjects pulling it up.

**This does not invalidate the run.** The contamination shifts the operating point rather than the
ordering, so the comparative results below hold: screening at 0.2 versus 0.4, N=50 versus N=300, the
global-versus-local divergence, and the reliability-beats-N conclusion are all relative contrasts
measured within the same calibration. Treat the **absolute** numbers (the fitted σ=0.35, dispersion
0.30, the noise map, and the specific asymptote values) as provisional pending a clean recalibration.

`load_pilot_subjects` now filters on `cohort` and `shine_variant` explicitly, and the stage-1 GT
script asserts it resolves exactly 41 pre-SHINE subjects, so this cannot recur silently.

### ⚠️ Required N was NOT determined

**In all 144 cells of `out/plateau_by_df_tr.csv`, `plateau_num_subjects == max_num_subjects == 300`,
on both the Spearman and the Procrustes M² curve.** The convergence curve never flattened inside the
swept range. Required-N is therefore bounded only as **>= 300** and is otherwise unknown.

This is easy to misread, because the file has a column literally named `plateau_num_subjects` full of
plausible values. It is reporting the ceiling, not a plateau. Any required-N quoted from this run is
a floor.

### Screening works, and is expensive

Averaged over the grid; cost from `out/tr24/coverage.csv`:

| `min_reliability` | Spearman asymptote ↑ | Procrustes M² ↓ | pass rate | candidates screened to retain 121 | retained mean TR |
|---|---|---|---|---|---|
| −1 (none) | 0.661 | 0.633 | 100% | 121 | 0.112 |
| 0.0 | 0.722 | 0.581 | 59.1% | 207 | 0.168 |
| 0.2 | 0.844 | 0.428 | 13.1% | 930 | 0.316 |
| 0.4 | **0.914** | **0.281** | **1.9%** | **6,283** | 0.484 |

Screening at 0.4 nearly halves M² but needs ~52x the recruitment. **0.2 gets most of the gain at
about one seventh the cost of 0.4** and looks like the sensible operating point.

Between-cohort Spearman at the pilot-realistic target (tr24, ndim=8, dispersion=0.3):

| N | none | 0.0 | 0.2 | 0.4 |
|---|---|---|---|---|
| 30 | 0.157 | 0.219 | 0.466 | 0.654 |
| 50 | 0.239 | 0.312 | **0.597** | **0.760** |
| 75 | 0.299 | 0.414 | 0.675 | 0.807 |
| 300 | 0.510 | 0.608 | 0.808 | 0.913 |

So R > 0.5 at N=50 is reachable, but only with screening at 0.2 or stricter.

### The closest-pair structure barely reproduces

Same cells, chance-corrected top-5% Jaccard (raw chance = 0.026):

| N | none | 0.0 | 0.2 | 0.4 |
|---|---|---|---|---|
| 30 | 0.011 | 0.017 | 0.041 | 0.071 |
| 300 | 0.058 | 0.072 | 0.131 | **0.232** |

**This is the central tension in the run.** At N=300 with the strictest screening, global Spearman is
0.913 while closest-pair agreement is 0.232. The global picture looks excellent while the local
structure, which is what stimulus construction depends on, is barely reproducible. The closest pairs
are precisely the near-ties, where small distance differences flip set membership; global Spearman is
carried by the many easy far pairs.

### Other gradients

- **Target reliability helps a lot**: asymptote 0.708 at TR 0.24 rising to 0.862 at TR 0.50.
- **Dimensionality barely moves Spearman** (0.78 flat from ndim 5 to 10) **but degrades M²**
  (0.428 to 0.532). Fit quality unchanged while cross-cohort generalisation worsens is the signature
  of over-parameterisation.

---

## 5. Two results that were revised

Recorded because both were briefly believed and would have misled if acted on.

### The GT's apparent high rank was an artifact

The pilot-derived GT looked near-isotropic (4.9% to 9.5% variance per dimension across 15), with an
effective rank of 213 and 216 dimensions needed for 90% variance, suggesting the pilot RDM was
noise-dominated with no recoverable structure.

**That evidence is withdrawn.** It was computed on the **mean-imputed** aggregate, where 63.6% of
pairs are filled with one constant. Filling asserts those pairs are all equidistant, and *k* mutually
equidistant points form a simplex needing *k*−1 dimensions, so the fill manufactures rank. A control
settles it: a synthetic **rank-8** space put through the identical mask and fill reports effective
rank **193** and needs **239** dimensions for 90% variance, statistically indistinguishable from the
real data's 213 and 216.

What survives is the split-half reliability of 0.26, computed on observed pairs only with no
imputation. The dimensionality-selection rule that depended on the spectrum has been removed.

### Top-k Jaccard understates usable structure

The 0.232 figure above is real, but it answers a stricter question than the stimulus-construction
goal poses. If three images form one tight cluster, which *pair* among them is "closest" flips with
noise, and Jaccard counts every flip as disagreement even though all of them support the same
practical decision: use one of the three. It penalises exactly the within-cluster reshuffling that is
irrelevant.

Neither Spearman (too global, dominated by far pairs) nor top-k Jaccard (too local, penalises
harmless flips) measures the actual decision. That is why the cluster-agreement metrics were added.

---

## 5b. task-v5 stage 2: the design comparison on a bounded canvas

The current run. 17,280 fits (1,728 configurations x 10 cohorts), GT at D=8 over the pre-SHINE pilot,
noise and dispersion fitted to all 47 pilot sessions. Report: `sim_results/design-comparison-v5/report_v5.html`.

### The designed allocation wins, on every embedding-level measure, at every N

Both arms collect identical numbers of judgements and differ only in which pairs receive them.

| | N=30 | N=50 | N=75 | N=500 |
|---|---|---|---|---|
| pair coverage, designed / random (%) | 38.4 / 32.5 | 60.3 / 48.1 | 81.3 / 62.6 | 100 / 99.9 |
| relative coverage gain | +17.9% | +25.4% | +29.9% | +0.1% |
| embedding agreement | .400 / .382 | .514 / .495 | .590 / .574 | .874 / .865 |
| Procrustes m2 (lower better) | .830 / .842 | .752 / .766 | .679 / .697 | .287 / .305 |
| top-k Jaccard | .142 / .136 | .179 / .172 | .213 / .205 | .452 / .437 |
| recovery AUC | .784 / .774 | .839 / .831 | .872 / .865 | .958 / .956 |

At N=50, paired within every sensitivity setting, all four embedding metrics favour the design in
96-99% of settings with Cohen's dz between 1.4 and 2.2. The deployable per-session constraint costs
**0.3% relative** coverage; the design also uses the image set ~5x more evenly and wastes ~9x fewer
judgements.

### ⚠️ Coverage is bought with per-pair precision, and only the embedding resolves it

Pre-MDS reliability - the Spearman between two cohorts' pooled distance matrices - is **lower** for
the designed arm through the deployable range (.126 vs .143 at N=50, favouring random in 96% of
settings). Same effort over more pairs means fewer observations each.

That reverses at the embedding. Weighted MDS uses the observation counts, so a pair measured once
still constrains the solution, and constraining more of the space beats constraining less of it more
precisely. **The pre-MDS correlation is the input to the method, not its output, and the design
decision should not rest on it.**

### ⚠️ The space does not support fine-grained clusters

All **5,184 configurations** (1,728 x 3 linkages) selected `k*=2`, on both the reproducibility (VI,
one-SE) and separation (cross-cohort silhouette) criteria - rules built to disagree where real
structure exists. Cross-cohort silhouette **crosses zero at k≈12** and is negative beyond it: points
sit closer to a neighbouring cluster than their own.

HDBSCAN corroborates independently. At `min_cluster_size=5`, 61% of images are unclustered and two
cohorts agree on *which* at κ=0.20. At `min_cluster_size=2` it finds 171 clusters that two cohorts
agree on at ARI 0.05. No setting both covers the image set and reproduces.

**Implication: deduplicate by a distance threshold, not by cluster membership.**

### ⚠️ How much of that is the ground truth rather than the method

`gt_diagnostics` compares the D=8 GT against the raw aggregate it was fitted from, with a half-split
noise ceiling as the control. The GT's within-level agreement runs 0.44-0.50 for the coarse levels
against ceilings of 0.07 / 0.02 / 0.13 / 0.41 - **3.9x to 20.7x the agreement the data reaches with
itself.** That is the signature of an embedding fitting noise, and it matches the pilot's own
out-of-sample split-half peaking at 0.233.

So the fine-grained nulls are partly a statement about 41 participants at 31% coverage, not about
SpAM or MDS. The *design* comparison is unaffected: both arms face the identical GT.

### The bounded canvas reproduces an observable it was never fitted to

RMSE between a participant's two judgements of a pair, against how far apart they placed it, is an
inverted U in the pilot. The high-distance turnover requires a bounded canvas. The simulation
reproduces it: `drop_from_peak` 0.27 against the pilot's 0.37, peak in the same region, `is_inverted_u`
true for both. The residual mismatch is at the **low** end - the model never gets as quiet on
obviously-similar pairs as people do, the same fine-grained weakness the semantic gradient shows from
the other side.

### ⚠️ Required N is still not determined

Recall of the GT's closest pairs is 0.50 at N=500 and AUC is still climbing; no curve plateaued. The
run locates where the *design advantage* lives (N=50-75, where the coverage gain peaks) and not where
recovery saturates. This has now been left open by two consecutive sweeps.

### Process failures worth remembering

* A resume that did not recognise completed work appended **449 duplicate fits**, entering the pair
  loop as self-comparisons (VI exactly 0) and biasing every rep-pair metric upward in the 48 affected
  cells - all at N=30, dispersion=0.1. Fixed in `_grouped_successful`; the four store-derived tables
  had to be recomputed.
* `k_selection.csv` was built by merging against a non-unique lookup, turning 144 groups into 186,624
  rows. k* is now chosen per full configuration and the merges assert `one_to_one`.
* The semantic-gradient check scored **one configuration per arm** (whichever came first in dict
  order) while reporting a conclusion about the cohorts. Now scored on every cell; the figure in the
  current report predates the fix.

## 6. What is not established

1. **Required N.** The sweep hit its ceiling; the answer is only "> 300". Extending the range would
   settle it, though the shallow slope in section 3 suggests reliability is the better lever.
2. **Cluster reproducibility.** The metrics exist but no run has produced them yet. This is the
   quantity closest to the actual goal.
3. **Designed vs random image-to-trial allocation.** Every result above came from the random arm.
   As sampling plans the designed arm is clearly better (at N=75, 81.3% vs 62.6% pair coverage, and
   3 to 5x tighter per-image balance), but whether that survives into recovered structure is
   untested.
4. **Whether the perceptual space is clustered at all.** If it is a continuum, "one image per
   cluster" is the wrong rule and a distance threshold should replace it. The diagnostics to decide
   this are implemented but unrun.
5. **Statistical power of the existing cells.** 6 reps gives 15 cohort pairs, and those pairs are not
   independent (each cohort appears in 5), so every reported SEM understates the true uncertainty.
6. **A clean recalibration.** The v4-fitted constants come from a set that included 14
   production subjects (section 4). Re-running stage 1 on the 41 pre-SHINE pilot subjects would put
   the absolute numbers on the footing the comparative ones already have.

## 7. Practical implications

- **Screen at `min_reliability` ≈ 0.2.** Most of the quality gain at roughly one seventh the
  recruitment cost of 0.4.
- **Prefer data quality over sample size.** N=30 at R=0.65 beats N=300 at R=0.24 for closest-pair
  recovery. Anything that raises per-subject reliability outperforms recruiting more people.
- **Do not trust a global agreement number for a local decision.** 0.913 global against 0.232
  closest-pair in the same cell.
- **Fit fewer dimensions rather than more.** Extra dimensions did not improve rank agreement and did
  worsen configuration agreement.
- **Sample the pair space deliberately.** Coverage broadens rather than deepens under random
  allocation, which is why per-pair reliability is stuck near 0.25 regardless of N.
