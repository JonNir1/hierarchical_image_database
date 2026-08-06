"""Tests for the bounded sort canvas.

The load-bearing property is that arrangements become geometrically possible: on an unbounded plane
the model produced per-trial maximum distances of ~1.39 when the canvas diagonal is 1.0 by
construction. Several tests below assert bounds that the previous model violated.
"""
import numpy as np
import pytest

from SpAM_Simulations import canvas as cv


def _square(n=12, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 2))


# --------------------------------------------------------------------- spec

def test_diagonal_and_upper_match_the_deployed_normalisation():
    spec = cv.CanvasSpec(aspect=0.75)
    assert spec.diagonal == pytest.approx(np.hypot(1.0, 0.75))
    np.testing.assert_allclose(spec.upper, [1.0, 0.75])


def test_spec_rejects_impossible_geometry():
    for bad in (cv.CanvasSpec(aspect=0.0), cv.CanvasSpec(aspect=1.5)):
        with pytest.raises(ValueError, match="aspect"):
            bad.validate()
    for bad in (cv.CanvasSpec(fill=0.0), cv.CanvasSpec(fill=1.5)):
        with pytest.raises(ValueError, match="fill"):
            bad.validate()


def test_defaults_come_from_the_pilot():
    """Not round numbers: the aspect is the pilot's median screen shape."""
    assert cv.DEFAULT_ASPECT == pytest.approx(0.499)      # PILOT-COHORT median screen shape
    assert cv.DEFAULT_FILL == pytest.approx(1.0)          # with soft walls pulling the edge in
    assert cv.DEFAULT_SOFTNESS == pytest.approx(4.0)      # matches the near-wall density


# --------------------------------------------------------------------- fitting

def test_fit_to_canvas_centres_and_fills_the_box():
    spec = cv.CanvasSpec(aspect=0.5, fill=1.0)
    Y = cv.fit_to_canvas(_square(), spec)
    assert np.all(Y >= -1e-9) and np.all(Y <= spec.upper + 1e-9)
    # Centred on the BOUNDING-BOX midpoint, not the centroid - see the comment in fit_to_canvas.
    np.testing.assert_allclose(0.5 * (Y.max(axis=0) + Y.min(axis=0)), spec.upper / 2, atol=1e-9)
    # The tighter axis fills exactly; the other may be looser, since the scale is isotropic.
    span = Y.max(axis=0) - Y.min(axis=0)
    assert np.isclose(span / spec.upper, 1.0, atol=1e-9).any()


def test_isotropic_mode_preserves_relative_geometry():
    """With `isotropic=True` one scale applies to both axes, so shape is untouched."""
    from scipy.spatial.distance import pdist
    Y = _square(seed=3)
    fitted = cv.fit_to_canvas(Y, cv.CanvasSpec(aspect=0.6, isotropic=True))
    ratio = pdist(fitted) / pdist(Y)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_the_default_per_axis_fit_uses_both_canvas_dimensions():
    """The pilot shows subjects filling width AND height; an isotropic fit cannot on a wide canvas."""
    spec = cv.CanvasSpec(aspect=0.494, fill=1.0)
    Y = cv.fit_to_canvas(_square(30), spec)
    span = Y.max(axis=0) - Y.min(axis=0)
    np.testing.assert_allclose(span, spec.upper, atol=1e-9)      # both axes filled

    iso = cv.fit_to_canvas(_square(30), spec._replace(isotropic=True))
    iso_span = iso.max(axis=0) - iso.min(axis=0)
    assert iso_span[0] < 0.75 * spec.upper[0]     # the short axis binds; width is left unused


def test_per_axis_scaling_distorts_geometry_and_that_is_intended():
    """A wide canvas stretches horizontal separations more than vertical ones. Modelled, not error."""
    from scipy.spatial.distance import pdist
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])           # right isoceles: two equal legs
    fitted = cv.fit_to_canvas(Y, cv.CanvasSpec(aspect=0.5, fill=1.0))
    d = pdist(fitted)
    assert d[0] > d[1]                                            # the horizontal leg is stretched


def test_fill_below_one_leaves_a_margin():
    spec = cv.CanvasSpec(aspect=0.5, fill=0.8)
    Y = cv.fit_to_canvas(_square(), spec)
    span = Y.max(axis=0) - Y.min(axis=0)
    assert np.all(span <= 0.8 * spec.upper + 1e-9)


def test_fit_to_canvas_survives_a_degenerate_arrangement():
    """All points coincident: no scale is defined, and dividing by the zero span must not happen."""
    Y = cv.fit_to_canvas(np.zeros((5, 2)), cv.CanvasSpec())
    assert np.isfinite(Y).all()
    np.testing.assert_allclose(Y, np.tile(cv.CanvasSpec().upper / 2, (5, 1)))


def test_a_skewed_arrangement_still_lands_inside_the_box():
    """The bug the bounding-box centring fixes: centring on the centroid put points outside.

    One far outlier drags the extent while leaving the centroid inside the cluster, so a
    centroid-centred fit hangs off the edge even at fill=1.
    """
    spec = cv.CanvasSpec(aspect=0.5, fill=1.0)
    Y = np.vstack([np.zeros((19, 2)) + 0.01 * np.arange(19)[:, None], [[50.0, 50.0]]])
    fitted = cv.fit_to_canvas(Y, spec)
    assert np.all(fitted >= -1e-9) and np.all(fitted <= spec.upper + 1e-9)


def test_fit_to_canvas_rejects_non_2d_input():
    with pytest.raises(ValueError, match=r"must be \(n, 2\)"):
        cv.fit_to_canvas(np.zeros((5, 3)))


# --------------------------------------------------------------------- placement

def test_place_never_leaves_the_box():
    """The ceiling itself. Enormous noise must still produce a legal arrangement."""
    spec = cv.CanvasSpec(aspect=0.5)
    rng = np.random.default_rng(0)
    Y = cv.place(cv.fit_to_canvas(_square(40), spec), noise=5.0, rng=rng, spec=spec)
    assert np.all(Y >= 0.0) and np.all(Y <= spec.upper)


def test_place_with_zero_noise_still_applies_the_soft_wall():
    """Not the identity, and deliberately so: the soft wall is part of how an arrangement renders.

    At fill=1.0 the fitted extremes sit on the boundary, and the wall pulls them in whether or not
    any placement error was added. That is what makes the simulated max distance 0.78 rather than
    the geometric 1.0.
    """
    spec = cv.CanvasSpec()
    Y = cv.fit_to_canvas(_square(), spec)
    placed = cv.place(Y, 0.0, np.random.default_rng(0), spec)
    np.testing.assert_allclose(placed, cv.soft_bound(Y, spec))
    assert not np.allclose(placed, Y)                      # the wall did something
    np.testing.assert_allclose(placed, cv.place(Y, 0.0, np.random.default_rng(9), spec))


def test_place_rejects_negative_noise():
    with pytest.raises(ValueError, match="non-negative"):
        cv.place(np.zeros((3, 2)), -0.1, np.random.default_rng(0))


# --------------------------------------------------------------------- distances

def test_canvas_distances_are_bounded_by_one():
    """The deployed normalisation: 1.0 is attainable only corner to corner."""
    spec = cv.CanvasSpec(aspect=0.5)
    corners = np.array([[0.0, 0.0], [1.0, 0.5]])
    assert cv.canvas_distances(corners, spec)[0] == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    d = cv.canvas_distances(cv.place(cv.fit_to_canvas(_square(30), spec), 0.3, rng, spec), spec)
    assert d.max() <= 1.0 + 1e-12


def test_arrange_produces_only_possible_distances():
    """The regression that motivated the module: the unbounded model reached 1.39."""
    spec = cv.CanvasSpec()
    rng = np.random.default_rng(2)
    maxima = [cv.arrange(_square(20, seed=s), 0.1, rng, spec).max() for s in range(50)]
    assert max(maxima) <= 1.0
    assert np.median(maxima) > 0.5      # and it does use the canvas, rather than hugging the centre


def test_max_distance_stats_flags_impossible_arrangements():
    ok = [np.array([0.5, 0.8]), np.array([0.7])]
    assert cv.max_distance_stats(ok)["frac_above_one"] == 0.0
    bad = [np.array([1.39]), np.array([0.8])]
    assert cv.max_distance_stats(bad)["frac_above_one"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no distance vectors"):
        cv.max_distance_stats([])


def test_simulated_max_distance_lands_near_the_pilots():
    """Calibration target: the pilot's per-trial max distance has median 0.802."""
    spec = cv.CanvasSpec()
    rng = np.random.default_rng(3)
    vecs = [cv.arrange(_square(20, seed=s), 0.12, rng, spec) for s in range(200)]
    stats = cv.max_distance_stats(vecs)
    assert stats["frac_above_one"] == 0.0
    # Pilot median is 0.802; DEFAULT_FILL is calibrated to land in this band.
    assert 0.70 < stats["median"] < 0.92, stats


# --------------------------------------------------------------------- ordering

def test_arrange_is_fit_then_place_then_normalise():
    """Pin the pipeline order: the scale is decided on the INTENDED arrangement, then perturbed.

    Renormalising the realised arrangement instead would let one item's motor error rescale every
    other item, which is not a thing a person does. The ordering is settled mechanistically: both
    orders were measured and NEITHER reproduces the empirical turnover (drop_from_peak 0.046
    fit-first, 0.066 fit-last, against 0.369 in the pilot), so the canvas is not the mechanism
    behind it and fit cannot arbitrate.
    """
    spec = cv.CanvasSpec()
    Y = _square(20, seed=7)
    expected = cv.canvas_distances(
        cv.place(cv.fit_to_canvas(Y, spec), 0.1, np.random.default_rng(11), spec), spec)
    actual = cv.arrange(Y, 0.1, np.random.default_rng(11), spec)
    np.testing.assert_allclose(actual, expected)


# --------------------------------------------------------------------- soft walls

def test_soft_bound_keeps_everything_strictly_inside():
    """Asymptotic, so even absurd input stays in the open box rather than landing on the wall."""
    spec = cv.CanvasSpec(aspect=0.5)
    Y = np.array([[-50.0, -50.0], [50.0, 50.0], [0.5, 0.25]])
    out = cv.soft_bound(Y, spec)
    assert np.all(out > 0.0) and np.all(out < spec.upper)


def test_soft_bound_barely_moves_the_interior():
    """Near-identity where it should be: the bound must not distort the middle of the canvas."""
    spec = cv.CanvasSpec(aspect=0.5, softness=4.0)
    half = spec.upper / 2
    quarter = half / 2                       # half way from centre to wall
    moved = cv.soft_bound(half + quarter, spec) - (half + quarter)
    assert np.all(np.abs(moved) / quarter < 0.02)


def test_soft_bound_leaves_the_centre_exactly_alone():
    spec = cv.CanvasSpec(aspect=0.5)
    np.testing.assert_allclose(cv.soft_bound(spec.upper / 2, spec), spec.upper / 2)


def test_infinite_softness_is_exactly_hard_clipping():
    """The family's limit, kept only so the rejected alternative can be measured against."""
    spec = cv.CanvasSpec(aspect=0.5, softness=float("inf"))
    Y = np.array([[-1.0, -1.0], [2.0, 2.0], [0.4, 0.2]])
    np.testing.assert_allclose(cv.soft_bound(Y, spec), np.clip(Y, 0.0, spec.upper))


def test_softer_walls_saturate_more_gently():
    spec = cv.CanvasSpec(aspect=0.5)
    at_wall = np.array([[1.0, 0.5]])
    soft = cv.soft_bound(at_wall, spec._replace(softness=2.0))
    hard = cv.soft_bound(at_wall, spec._replace(softness=16.0))
    assert np.all(soft < hard)               # gentler p pulls the point further in


def test_the_bound_produces_no_pile_up_at_the_wall():
    """The empirical constraint: 0.005% of pilot placements sit at the exact extreme.

    A hard clip puts 5.2% there. Anything that manufactures a point mass at the boundary is
    reproducing an artifact of the bounding rule rather than a property of the task.
    """
    spec = cv.CanvasSpec()
    rng = np.random.default_rng(5)
    placed = np.vstack([cv.place(cv.fit_to_canvas(_square(20, seed=s), spec), 0.08, rng, spec)
                        for s in range(150)])
    at_wall = np.mean((placed <= 0.0) | (placed >= spec.upper))
    assert at_wall == 0.0

    clipped = np.vstack([cv.place(cv.fit_to_canvas(_square(20, seed=s), spec), 0.08, rng,
                                  spec._replace(softness=float("inf")))
                         for s in range(150)])
    assert np.mean((clipped <= 0.0) | (clipped >= spec.upper)) > 0.01   # the artifact, for contrast


# --------------------------------------------------------------------- empirical sampling

def test_sample_spec_stays_within_the_observed_range():
    """Resampled from measured quantiles, so no draw should invent a screen the pilot never had."""
    rng = np.random.default_rng(0)
    specs = [cv.sample_spec(rng) for _ in range(500)]
    aspects = np.array([sp.aspect for sp in specs])
    fills = np.array([sp.fill for sp in specs])
    assert aspects.min() >= 0.463 - 1e-9 and aspects.max() <= 0.643 + 1e-9
    assert np.all((fills > 0) & (fills <= 1.0))
    assert aspects.std() > 0 and fills.std() > 0          # it really does vary


def test_sample_spec_recovers_the_missing_trial_to_trial_spread():
    """The reason it exists: a fixed spec makes every trial use the same extent, which is wrong.

    Pilot per-trial max distance has sd 0.106; a fixed spec gives 0.039. Sampling aspect and fill
    from the observed marginals should land much closer without adding a free parameter.
    """
    rng = np.random.default_rng(1)
    fixed, sampled = [], []
    for s in range(300):
        Y = _square(20, seed=s)
        fixed.append(cv.arrange(Y, 0.08, rng, cv.CanvasSpec()).max())
        sampled.append(cv.arrange(Y, 0.08, rng, cv.sample_spec(rng)).max())
    assert np.std(sampled) > 1.5 * np.std(fixed)
    assert max(sampled) <= 1.0                             # still geometrically possible


def test_sample_spec_can_hold_fill_fixed():
    rng = np.random.default_rng(2)
    specs = [cv.sample_spec(rng, vary_fill=False) for _ in range(50)]
    assert {sp.fill for sp in specs} == {cv.DEFAULT_FILL}
    assert len({sp.aspect for sp in specs}) > 1            # aspect still varies


def test_sample_spec_honours_the_softness_argument():
    """Softness is swept as a sensitivity axis, so it must not be overwritten by the sampler."""
    rng = np.random.default_rng(3)
    assert cv.sample_spec(rng, softness=6.0).softness == pytest.approx(6.0)
