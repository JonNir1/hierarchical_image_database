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
    assert cv.DEFAULT_ASPECT == pytest.approx(0.494)      # pilot median screen shape
    assert cv.DEFAULT_FILL == pytest.approx(0.85)         # calibrated to maxd median 0.802


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


def test_place_with_zero_noise_is_the_identity():
    Y = cv.fit_to_canvas(_square(), cv.CanvasSpec())
    np.testing.assert_allclose(cv.place(Y, 0.0, np.random.default_rng(0)), Y)


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

def test_scaling_must_precede_the_noise():
    """Measured: renormalising the REALISED arrangement destroys the ceiling effect entirely.

    The turnover exists because a point pinned against a wall can only move inward. If the
    arrangement is rescaled *after* the noise, the extreme points set the scale, their own jitter
    propagates into it, and the truncation cancels - `drop_from_peak` measured 0.000 that way. This
    asserts the ordering `arrange` uses is the one that keeps a turnover at all.
    """
    from SpAM_Simulations import validity as val
    spec = cv.CanvasSpec()

    def curve(fit_last: bool):
        oa, ra = [], []
        for s in range(250):
            Y = _square(20, seed=s)
            rng = np.random.default_rng(1000 + s)
            out = []
            for _ in range(2):
                if fit_last:
                    noisy = Y + rng.normal(0, 0.12 * np.ptp(Y), Y.shape)
                    out.append(cv.canvas_distances(cv.fit_to_canvas(noisy, spec), spec))
                else:
                    out.append(cv.arrange(Y, 0.12, rng, spec))
            oa.append(out[0]); ra.append(out[1])
        return val.noise_curve_shape(
            val.noise_vs_distance(np.concatenate(oa), np.concatenate(ra), n_bins=10))

    assert curve(fit_last=False)["drop_from_peak"] > curve(fit_last=True)["drop_from_peak"]
