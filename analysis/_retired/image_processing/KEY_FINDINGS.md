# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.

Did SHINE equalize the 725 stimuli, and did it do so equally across categories? Data: `images/pre_shine` and
`images/post_shine`, 725 images each (298 animate, 427 inanimate; L2 groups: handmade_object 267,
natural_object 160, human_face 102, animal_body 81, human_body 62, animal_face 53). Five features per image
(mean luminance, Michelson and RMS contrast, global contrast factor, perimeter-to-area ratio of the alpha mask),
foreground pixels only. Tests are the original parametric ones (t-test, one-way ANOVA, Bonferroni t-tests,
Tukey HSD); `analysis/utils/stats.py` has only rank-based comparisons, so they were not swapped. Bars show a
seeded 1,000-draw bootstrap 95% CI of the mean. Numbers come from a sandbox re-run on 2026-11-24, whose
output is now the notebook's stored output (it replaced the older matplotlib run; the one printed number that
changed is the sub-sampled pixel correlation, below).

- **Caveat, SHINE itself is not re-run:** the MATLAB pipeline (`resize_images.m`, `shine_equalize.m`) was not
  executed in this review. Its unit tests (`__tests__/TestShineEqualize.m`, 18 tests) pass under MATLAB R2023b
  with the corrected CI path.

## Did SHINE equalize luminance?

Notebook: [`postprocess_analysis.ipynb`](postprocess_analysis.ipynb) (commit `5562e37f`)

- **Spread collapsed:** mean luminance goes from 0.592 (SD 0.169 across images) to 0.579 (SD 0.0012).
- **By group:** every L2 group lands within 0.5791 to 0.5797 after SHINE, from 0.499 (human_face) to 0.680 (natural_object) before.
- **Pixel similarity left:** for 1,000 randomly sampled post-SHINE image pairs (seed 42) the mean RGB Pearson correlation is 0.267 (SD 0.165). The earlier stored output said 0.2678; the images or their order had changed since it was run.

## Are the post-SHINE images still different across categories?

Notebook: [`postprocess_analysis.ipynb`](postprocess_analysis.ipynb) (commit `5562e37f`)

- **Animate vs inanimate (t-test, post-SHINE):** mean luminance p < 0.001, RMS contrast p < 0.001, perimeter ratio p < 0.001; Michelson contrast p = 0.130 and global contrast factor p = 0.267 do not differ.
- **Across the six L2 groups (ANOVA, post-SHINE):** all features differ at p < 0.001 except Michelson contrast (p = 0.615). Many Bonferroni-corrected pairs differ, especially animal_face against the rest on global contrast factor and perimeter ratio.
- **Caveat, tiny absolute differences:** post-SHINE luminance differs by 0.0003 between animate and inanimate, so p < 0.001 there reflects a small SD, not a visible gap.
- **Perimeter ratio** depends only on the alpha mask, which SHINE should not touch; its category differences are properties of the objects.

## Did SHINE change categories differently?

Notebook: [`postprocess_analysis.ipynb`](postprocess_analysis.ipynb) (commit `5562e37f`)

- **Delta (post minus pre) by animate vs inanimate:** all four features differ at p < 0.001 (perimeter ratio is excluded, because SHINE should not change it).
- **Delta across L2 groups:** ANOVA p < 0.001 for all four features, with 8 to 9 of the 15 Tukey pairs below 0.05 per feature. So SHINE was not category-neutral: it moved some groups more than others (images that started further from the common luminance moved further).
- **Caveat:** this is the expected consequence of equalizing a feature that differed across categories before; the notebook does not test whether the residual differences matter for the SpAM task.
