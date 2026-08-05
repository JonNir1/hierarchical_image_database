# SpAM simulation findings

What the simulation runs have established, what they have not, and which conclusions were later
withdrawn. Every number here is read from a committed result file or recomputed from `data/`; the
source is named so each can be re-derived.

For *how* the pipeline works see [README.md](README.md), and for how to run it
[Cookbook.md](Cookbook.md).

> **Read the caveats before quoting anything.** Two results here were revised after the fact, the
> headline required-N figure is a **non-result** (the sweep never converged), and the latest run's
> calibration set included 14 production subjects alongside the pilot. All three are flagged inline
> rather than left for a reader to notice. None of them overturns the comparative findings.

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
| **task-v4-fitted** | + noise *shape* fitted, not just scale | the current headline run |

Tag `spam-sim-v4.0` (`53d667c`) is the revision the current `sim_results/` were produced from.

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

## 4. task-v4-fitted: the current run

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
6. **A clean recalibration.** The current fitted constants come from a set that included 14
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
