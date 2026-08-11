"""The 2-D sort canvas: the bounded rectangle subjects actually arrange images on.

Every simulation up to task-v4 placed points on an **unbounded** plane.
``task_v3_experiment``'s docstring recorded the omission ("a fixed-canvas ceiling/saturation is a
documented future refinement, not modelled") but the consequence was not measured until now, and it
is not small: the unbounded model produces a median per-trial maximum distance of **1.39**, when the
canvas diagonal is 1.0 by construction and the real pilot sits at **0.80**. The simulation was
generating arrangements that cannot exist.

**What the deployed task does** (``SpAM_Task/js/utils.js::computePairwiseDistances``): images are
placed in a rectangle whose top-left is (0, 0) and bottom-right is ``(sort_area_width,
sort_area_height)``, and every pairwise distance is divided by that rectangle's diagonal. So the
observable is in [0, 1] with 1 attainable only corner-to-corner.

**Measured on the PILOT COHORT ONLY** (26 v3+ participants, 440 non-catch trials):

======================================  ==========================================
aspect ``height / width``               median 0.499, quartiles [0.474, 0.540]
per-trial max normalised distance       median 0.779, sd 0.106, quartiles [0.718, 0.815]
placements at the exact wall            x 0.023%, y 0.034% (of 8,800 each)
======================================  ==========================================

**Cohort basis, and why it is narrower than it looks.** The same policy that governs the ground
truth applies here: production data must not calibrate a simulation whose purpose is to plan the
production study. An earlier version of these constants was measured over 114 v3+ participants
without a cohort filter, 88 of whom were production - the same contamination already recorded for
the task-v4 run. Corrected, the base is 26 subjects, because canvas geometry needs trial-level
``final_locations`` that pre-v3 sessions do not carry. That is thinner than the 41 subjects behind
the ground truth or the 47 behind the noise fit, and any report of these constants should say so.

``aspect`` turns out to be cohort-invariant (0.494 over everyone against 0.499 over the pilot),
which is what one expects of screen geometry, but the pilot value is used anyway rather than
arguing for an exception to the policy.

Two things follow. Subjects **do** spread out to use the canvas rather than clustering in the
middle, so a trial's arrangement is scaled to occupy the box. And the max distance is tightly
concentrated well below the 1.0 ceiling, which is what :data:`DEFAULT_FILL` and
:data:`DEFAULT_SOFTNESS` are jointly calibrated against.

**The walls are soft, and that is measured rather than assumed.** Across the pilot cohort's 8,800
placements per axis, only 0.023% (x) and 0.034% (y) sit at the exact canvas extreme, so there is no
point mass at the boundary - but there *is* a pile-up just inside it, with a sharp dead zone in the
outermost band. ``np.clip`` reproduces a spike where the data has none, putting 5.2% of placements
exactly on a wall. Per-axis min-max rescaling is worse still: it pins one item to each end of each
axis every trial, i.e. ~20% of placements.
:func:`soft_bound` reproduces the observed shape, and it also recovers far more of the empirical
noise-curve turnover than clipping does (``drop_from_peak`` 0.21 against 0.02, versus 0.37 in the
pilot), because it compresses over a region instead of only at the exact edge.

**Scaling happens before the noise, on mechanistic grounds rather than empirical ones.** The scale
is a decision the subject makes about the arrangement they intend - "spread these across the space"
- and the placement error then applies to each individual drag. Renormalising the *realised*
arrangement would instead let one item's motor error rescale every other item, which is not a thing
a person does. Both orders were measured and neither reproduces the empirical turnover
(``drop_from_peak`` 0.046 fit-first against 0.066 fit-last, versus 0.369 in the pilot), so the
ordering is settled by the mechanism, not by fit. An earlier note here claimed fit-last collapsed
the turnover to 0.000; that was measured under isotropic scaling and does not hold for the per-axis
fit this module actually uses.

**Noise becomes absolute.** On an unbounded plane the jitter had to be expressed relative to each
trial's own spread, since there was no other scale. With a fixed canvas there is one: motor and
decision precision are properties of the screen, not of the arrangement. So ``noise`` here is a
fraction of the canvas *width*, and ``subjects_noise_scale`` therefore means something different
from the v3/v4 constant of the same name. **Every calibrated constant must be re-derived**; the old
values are not transferable.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from scipy.spatial.distance import pdist

# Median aspect across the 26 pilot-cohort v3+ participants (quartiles [0.474, 0.540]). Screen
# shapes vary, so this is a representative canvas rather than any one subject's; `sample_spec`
# draws from the empirical distribution instead when per-subject variation is wanted.
DEFAULT_ASPECT = 0.499
# Fraction of the box a trial's arrangement is scaled to occupy before the walls soften it. Full
# extent, because `soft_bound` then pulls the periphery in: at fill=1.0 with softness 4 and
# placement noise 0.08 the simulated maximum distance has median 0.783 against the pilot's 0.779.
# (Under the rejected hard clip the equivalent value was 0.85; the two are not comparable.)
#
# NOTE: with a FIXED spec the simulated max distance is far too consistent across trials - sd 0.039
# against the pilot's 0.106 - because `fit_to_canvas` then scales every trial to the same extent.
# `sample_spec` draws aspect and fill per subject/trial from the observed marginals and recovers
# most of it (sd 0.089), which is why the sweep should use it rather than the bare default.
DEFAULT_FILL = 1.0
# Saturation exponent. The pilot shows essentially no mass at the wall (0.02-0.03% of placements sit
# at the exact extreme) but a pile-up just inside it, so the bound must be smooth, not a clip. p=4
# leaves the interior essentially untouched - a point half way to the wall moves by 0.8% - while
# saturating anything pushed past it.
DEFAULT_SOFTNESS = 4.0


class CanvasSpec(NamedTuple):
    """A sort rectangle of width 1 and height ``aspect``, with distances in canvas-diagonal units.

    ``isotropic=False`` (the default) scales each axis independently, which is what the pilot shows
    subjects doing: the observed extent is about 0.9 of the width *and* 0.8 of the height, i.e. both
    dimensions get used. An isotropic fit cannot do that on a canvas twice as wide as it is tall -
    the short axis binds and the arrangement sits in a square puddle in the middle, giving a
    per-trial max distance of 0.63 against the pilot's 0.80.

    The consequence is deliberate and worth stating: per-axis scaling **distorts** the arrangement's
    relative geometry, stretching horizontal separations more than vertical ones. That is not an
    approximation error, it is what a wide canvas does to a SpAM judgement, and leaving it out is
    what made the old model's distances unreachable. ``isotropic=True`` preserves shape exactly and
    is kept for the comparison.
    """
    aspect: float = DEFAULT_ASPECT
    fill: float = DEFAULT_FILL
    isotropic: bool = False
    # Exponent of the smooth saturation at the walls (see `soft_bound`). Larger is harder;
    # `float("inf")` is exactly `np.clip`, which the pilot's placement density rules out.
    softness: float = DEFAULT_SOFTNESS

    @property
    def diagonal(self) -> float:
        """Corner-to-corner distance, the divisor that puts observed distances in [0, 1]."""
        return float(np.hypot(1.0, self.aspect))

    @property
    def upper(self) -> np.ndarray:
        """Bottom-right corner, i.e. the per-axis upper bound for a placement."""
        return np.array([1.0, self.aspect], dtype=np.float64)

    def validate(self) -> None:
        if not 0 < self.aspect <= 1:
            raise ValueError(f"`aspect` must be in (0, 1], got {self.aspect}")
        if not 0 < self.fill <= 1:
            raise ValueError(f"`fill` must be in (0, 1], got {self.fill}")
        if self.softness <= 0:
            raise ValueError(f"`softness` must be positive, got {self.softness}")


# Empirical marginals from the 26 pilot-cohort v3+ participants, for per-subject/per-trial sampling.
# Quantiles rather than a fitted parametric family: with n=26 subjects there is no basis for
# choosing a distribution, and resampling the observed values assumes nothing.
_ASPECT_QUANTILES = (0.463, 0.474, 0.499, 0.540, 0.643)
_MAX_DISTANCE_QUANTILES = (0.324, 0.718, 0.779, 0.815, 0.883)


def sample_spec(rng: np.random.Generator, softness: float = DEFAULT_SOFTNESS,
                vary_fill: bool = True) -> CanvasSpec:
    """Draw a canvas from the pilot's observed screen shapes, for per-subject heterogeneity.

    Fixing ``aspect`` and ``fill`` at their medians makes every simulated trial use exactly the same
    extent, which the data contradicts: the pilot's per-trial max distance has sd 0.106 while the
    fixed-spec simulation gives 0.039. Sampling recovers most of it, reaching 0.089 - close but not
    exact, and the residual is plausibly content-dependent spread (a trial of near-identical images
    presumably gets arranged more tightly) that this model has no mechanism for.

    Costs no free parameter: both marginals are resampled from measured quantiles rather than fitted.
    """
    aspect = float(np.interp(rng.random(), np.linspace(0, 1, len(_ASPECT_QUANTILES)),
                             _ASPECT_QUANTILES))
    fill = DEFAULT_FILL
    if vary_fill:
        # The observed max-distance spread, expressed as a multiplicative deviation about its median
        # and applied to `fill`; clipped to (0, 1] since a trial cannot use more than the canvas.
        draw = float(np.interp(rng.random(), np.linspace(0, 1, len(_MAX_DISTANCE_QUANTILES)),
                               _MAX_DISTANCE_QUANTILES))
        fill = float(np.clip(DEFAULT_FILL * draw / _MAX_DISTANCE_QUANTILES[2], 0.3, 1.0))
    return CanvasSpec(aspect=aspect, fill=fill, softness=softness)


def fit_to_canvas(Y: np.ndarray, spec: CanvasSpec = CanvasSpec()) -> np.ndarray:
    """Centre an arrangement and scale it to occupy ``spec.fill`` of the box.

    Per-axis by default (see :class:`CanvasSpec`), so both canvas dimensions get used as the pilot
    shows subjects using them; ``spec.isotropic`` instead applies the tighter of the two factors to
    both axes and preserves shape. A degenerate arrangement (all points coincident on an axis)
    falls back to no scaling on that axis rather than dividing by zero.
    """
    spec.validate()
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"`Y` must be (n, 2), got {Y.shape}")
    # Centre on the BOUNDING-BOX midpoint, not the centroid. The scale is derived from the span, so
    # centring on the mean would leave a skewed arrangement (one dense cluster plus a far outlier)
    # hanging outside the box even at fill=1: its centroid sits near the cluster while its extent is
    # measured from the outlier.
    centred = Y - 0.5 * (Y.max(axis=0) + Y.min(axis=0))
    span = centred.max(axis=0) - centred.min(axis=0)
    limits = spec.fill * spec.upper
    with np.errstate(divide="ignore", invalid="ignore"):
        per_axis = np.where(span > 0, limits / np.maximum(span, 1e-12), np.inf)
    if spec.isotropic:
        tightest = float(np.min(per_axis))
        per_axis = np.full(2, tightest if np.isfinite(tightest) else 1.0)
    # A collapsed axis has no scale of its own; leave it alone rather than blowing up.
    per_axis = np.where(np.isfinite(per_axis), per_axis, 1.0)
    return centred * per_axis + spec.upper / 2.0


def soft_bound(Y: np.ndarray, spec: CanvasSpec = CanvasSpec()) -> np.ndarray:
    """Smoothly saturate an arrangement into the box. Near-identity inside, asymptotic at the walls.

    ``u / (1 + |u/h|^p)^(1/p)`` about the canvas centre, where ``h`` is the half-extent and ``p`` is
    :attr:`CanvasSpec.softness`. The family interpolates between a gentle squash and a hard clip:
    ``p -> inf`` recovers ``np.clip``, while at the default ``p = 4`` a point half way to the wall
    moves by 0.8% and one at the wall lands at 0.84 of the half-extent.

    **Hard clipping is ruled out by the data, not by taste.** Measured over 44,080 pilot placements,
    only **0.005%** of items sit at the exact canvas extreme and **0.02%** within 1% of it, so there
    is no point mass at the wall; clipping manufactures one. What the pilot does show is a *pile-up
    just inside* the edge (the 5-10% band holds 2x the uniform density on x, the 10-15% band 2.7x on
    y) followed by a sharp dead zone in the outermost band - the signature of a soft wall plus the
    item's own half-width. This function reproduces that shape; ``np.clip`` reproduces a spike.

    **Per-axis min-max rescaling is ruled out for the same reason.** It pins exactly one item to
    each end of each axis every trial, which at 20 items per trial would put ~20% of placements
    exactly on a wall against the observed 0.02%.
    """
    spec.validate()
    Y = np.asarray(Y, dtype=np.float64)
    half = spec.upper / 2.0
    u = Y - half
    p = float(spec.softness)
    if not np.isfinite(p):                      # p = inf is the hard-clip limit, kept for comparison
        return np.clip(Y, 0.0, spec.upper)
    scaled = np.abs(u) / half
    return half + u / np.power(1.0 + np.power(scaled, p), 1.0 / p)


def place(Y: np.ndarray, noise: float, rng: np.random.Generator,
          spec: CanvasSpec = CanvasSpec()) -> np.ndarray:
    """Perturb a canvas-fitted arrangement by placement noise and bound it into the box.

    ``noise`` is an absolute fraction of the canvas width (see the module docstring). The bound is
    what produces the ceiling: a point already near a wall has room to move inward but not outward,
    so the upper tail of its distance to a far partner is compressed while the lower tail is not.
    Saturation is smooth (:func:`soft_bound`) rather than a clip, because the pilot shows no mass at
    the wall.
    """
    spec.validate()
    Y = np.asarray(Y, dtype=np.float64)
    if noise < 0:
        raise ValueError(f"`noise` must be non-negative, got {noise}")
    jittered = Y + rng.normal(0.0, noise, size=Y.shape) if noise > 0 else Y
    return soft_bound(jittered, spec)


def canvas_distances(Y: np.ndarray, spec: CanvasSpec = CanvasSpec()) -> np.ndarray:
    """Condensed pairwise distances in canvas-diagonal units, matching the deployed task.

    Mirrors ``SpAM_Task/js/utils.js::computePairwiseDistances``: Euclidean distance divided by
    ``sqrt(w^2 + h^2)``, so the result is in [0, 1] and 1 is attainable only corner-to-corner.
    """
    return pdist(np.asarray(Y, dtype=np.float64)) / spec.diagonal


def arrange(Y: np.ndarray, noise: float, rng: np.random.Generator,
            spec: CanvasSpec = CanvasSpec()) -> np.ndarray:
    """``fit_to_canvas`` then :func:`place` then :func:`canvas_distances`, the whole trial pipeline.

    The order is load-bearing: fit the *intended* arrangement, then perturb, then bound. Perturbing
    before fitting would let the noise set the scale and cancel the truncation (measured: it drops
    the noise curve's turnover to exactly zero).
    """
    return canvas_distances(place(fit_to_canvas(Y, spec), noise, rng, spec), spec)


def max_distance_stats(distance_vectors) -> dict:
    """Distribution of the per-trial maximum distance, the statistic ``fill`` is calibrated on.

    The pilot's is median 0.802, sd 0.082. It is the sharpest single check that a simulated
    arrangement is geometrically possible at all: the unbounded model scores 1.39 against a hard
    ceiling of 1.0.
    """
    maxima = np.array([np.max(d) for d in distance_vectors if np.size(d)], dtype=np.float64)
    if maxima.size == 0:
        raise ValueError("no distance vectors supplied")
    return {
        "median": float(np.median(maxima)), "mean": float(maxima.mean()),
        "sd": float(maxima.std(ddof=1)) if maxima.size > 1 else 0.0,
        "p05": float(np.percentile(maxima, 5)), "p95": float(np.percentile(maxima, 95)),
        "max": float(maxima.max()), "n_trials": int(maxima.size),
        # A value above 1 is impossible on a real canvas and means the bound is not being applied.
        "frac_above_one": float(np.mean(maxima > 1.0)),
    }


# --------------------------------------------------------------------------- trial simulator
def make_canvas_trial_simulator(spec: Optional[CanvasSpec] = None,
                                sample_per_trial: bool = True,
                                softness: float = DEFAULT_SOFTNESS):
    """A drop-in replacement for ``task_v3_experiment._simulate_trial`` that uses a bounded canvas.

    Returned as a closure with the *same* signature as the unbounded original, so it can be injected
    into the existing v3/v4 machinery rather than forking it. That matters: ``task_v4_experiment``
    already imports v3's trial simulator "rather than duplicated, so the two models cannot drift
    apart", and duplicating 400 lines to change one step would undo that.

    Differences from the unbounded original, all consequences of the canvas rather than choices:

    * the arrangement is scaled into the box before the noise, so distances are in canvas-diagonal
      units and cannot exceed 1;
    * ``subject_noise`` is an **absolute** fraction of canvas width, not a ratio to the trial's own
      spread, because a fixed canvas supplies the scale that an unbounded plane did not;
    * the walls saturate smoothly (:func:`soft_bound`) rather than clipping.

    ``sample_per_trial`` draws the canvas from the pilot's observed screen shapes for each trial
    (:func:`sample_spec`), which is what restores the per-trial variability in extent that a fixed
    spec loses. Pass an explicit ``spec`` to hold it fixed instead, e.g. when sweeping ``softness``.
    """
    from SpAM_Simulations.models.task_v3_experiment import _condensed_pair_indices, project_2d

    def simulate(trial_images, pair_rows, pair_cols, N, gt_embeddings, weights, subject_noise,
                 observations, n_obs, rng):
        this = spec if spec is not None else (
            sample_spec(rng, softness=softness) if sample_per_trial
            else CanvasSpec(softness=softness))
        coords = gt_embeddings[trial_images] * weights
        fitted = fit_to_canvas(project_2d(coords), this)
        placed = place(fitted, float(subject_noise), rng, this)
        trial_dists = canvas_distances(placed, this).astype(np.float32)
        cond_idx = _condensed_pair_indices(trial_images[pair_rows], trial_images[pair_cols], N)
        observations[cond_idx] += trial_dists
        n_obs[cond_idx] += 1
        return cond_idx, trial_dists, placed.astype(np.float32)

    return simulate
