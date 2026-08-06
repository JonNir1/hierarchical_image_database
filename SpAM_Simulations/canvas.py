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

**Measured from the pilot** (114 v3+ participants, 2,204 non-catch trials):

======================================  ==========================================
aspect ``height / width``               median 0.494, 5-95% [0.451, 0.643]
per-trial max normalised distance       median 0.802, sd 0.082, 5-95% [0.616, 0.876]
observed coordinate extent              x approx [0.0, 0.9], y approx [0.0, 0.8]
======================================  ==========================================

Two things follow. Subjects **do** spread out to use the canvas rather than clustering in the
middle, so a trial's arrangement is scaled to occupy most of the box. And the max distance is
tightly concentrated well below the 1.0 ceiling, so the scaling is close to, but not exactly, a full
fit - which is what :data:`DEFAULT_FILL` is calibrated against.

**Scaling happens before the noise, not after.** Both orders were measured. Scaling the *intended*
arrangement and then perturbing it reproduces a turnover in the noise-vs-distance curve;
renormalising the *realised* arrangement destroys it completely (``drop_from_peak`` falls to 0.000),
because the renormalisation is driven by the extreme points themselves, so their noise propagates
into the scale and cancels the very truncation it was supposed to create.

**Noise becomes absolute.** On an unbounded plane the jitter had to be expressed relative to each
trial's own spread, since there was no other scale. With a fixed canvas there is one: motor and
decision precision are properties of the screen, not of the arrangement. So ``noise`` here is a
fraction of the canvas *width*, and ``subjects_noise_scale`` therefore means something different
from the v3/v4 constant of the same name. **Every calibrated constant must be re-derived**; the old
values are not transferable.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial.distance import pdist

# Median aspect across the pilot's 114 v3+ participants. Screen sizes vary (5-95% is [0.451,
# 0.643]), so this is a representative canvas rather than any one subject's.
DEFAULT_ASPECT = 0.494
# Fraction of the box a trial's arrangement is scaled to occupy. Calibrated against the pilot's
# per-trial maximum distance: at fill=0.85 with placement noise 0.08 the simulated maximum has
# median 0.807 and sd 0.071, against the observed 0.802 and 0.082. Subjects leave a small margin
# rather than pushing items flush to the walls, which is why this is below 1.
DEFAULT_FILL = 0.85


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


def place(Y: np.ndarray, noise: float, rng: np.random.Generator,
          spec: CanvasSpec = CanvasSpec()) -> np.ndarray:
    """Perturb a canvas-fitted arrangement by placement noise and clip it into the box.

    ``noise`` is an absolute fraction of the canvas width (see the module docstring). Clipping is
    what produces the ceiling: a point already against a wall can be pushed inward but not outward,
    so the upper tail of its distance to a far partner is truncated while the lower tail is not.
    """
    spec.validate()
    Y = np.asarray(Y, dtype=np.float64)
    if noise < 0:
        raise ValueError(f"`noise` must be non-negative, got {noise}")
    jittered = Y + rng.normal(0.0, noise, size=Y.shape) if noise > 0 else Y
    return np.clip(jittered, 0.0, spec.upper)


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
