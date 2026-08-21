"""Tests for the RQ2 power machinery: GT perturbation, cross-store correlation, power."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from SpAM_Simulations.empirical.gt_perturbation import (
    build_perturbed_set, distance_similarity, perturb_gt,
)
from SpAM_Simulations.measures import rq2_power as rp


def _coords(n=80, d=5, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


class TestPerturbGt:
    @pytest.mark.parametrize("target", [0.99, 0.95, 0.90, 0.70])
    def test_hits_its_target(self, target):
        coords = _coords()
        out, info = perturb_gt(coords, target, seed=1)
        assert info["converged"], info
        assert abs(info["achieved_rho"] - target) <= 0.01
        assert abs(distance_similarity(coords, out) - target) <= 0.01

    def test_is_seed_reproducible(self):
        coords = _coords()
        a, _ = perturb_gt(coords, 0.95, seed=7)
        b, _ = perturb_gt(coords, 0.95, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_directions(self):
        """build_perturbed_set relies on this: one scaled direction would smooth the power curve."""
        coords = _coords()
        a, _ = perturb_gt(coords, 0.95, seed=1)
        b, _ = perturb_gt(coords, 0.95, seed=2)
        assert not np.allclose(a, b)

    def test_rho_of_one_is_the_identity(self):
        coords = _coords()
        out, info = perturb_gt(coords, 1.0, seed=0)
        np.testing.assert_array_equal(out, coords)
        assert info["noise_scale"] == 0.0

    def test_rejects_an_impossible_target(self):
        with pytest.raises(ValueError, match="target_rho"):
            perturb_gt(_coords(), 0.0)
        with pytest.raises(ValueError, match="target_rho"):
            perturb_gt(_coords(), 1.5)

    def test_similarity_falls_monotonically_with_the_effect_size(self):
        """The bisection assumes this; if it failed the search would not converge."""
        coords = _coords()
        # base seed 1, not 0: `_coords` also draws from default_rng(0), so seed=0 would hand the
        # first target a perturbation direction identical to the coordinates (see the degenerate
        # case below).
        built = build_perturbed_set(coords, (0.99, 0.95, 0.90), seed=1)
        achieved = [built[t][1]["achieved_rho"] for t in (0.99, 0.95, 0.90)]
        assert achieved[0] > achieved[1] > achieved[2]

    def test_the_target_is_on_distances_not_coordinates(self):
        """Two embeddings can differ in coordinates while inducing near-identical distances."""
        coords = _coords()
        out, _ = perturb_gt(coords, 0.95, seed=1)
        coord_corr = spearmanr(coords.ravel(), out.ravel()).statistic
        dist_corr = spearmanr(pdist(coords), pdist(out)).statistic
        assert abs(dist_corr - 0.95) < 0.01
        assert coord_corr != pytest.approx(dist_corr, abs=1e-6)

    def test_a_degenerate_direction_raises_instead_of_running_away(self):
        """Noise parallel to the coordinates is a pure rescale: every distance rank is unchanged,
        so the similarity stays at 1.0 at ANY scale and the bracket search would double forever.

        This is reachable: `_coords` and `perturb_gt` both draw from `default_rng`, so passing the
        same seed to each produces exactly this. It cost a debugging round to find, because the
        symptom was a noise scale of 1e17 rather than an error.
        """
        coords = np.random.default_rng(0).standard_normal((80, 5))
        with pytest.raises(ValueError, match="degenerate"):
            perturb_gt(coords, 0.95, seed=0)         # same seed -> noise IS the coordinates


def _draws(cells, values_by_cell, extra_col="rep"):
    rows = []
    for cell, values in zip(cells, values_by_cell):
        for i, v in enumerate(values):
            rows.append({"num_subjects": cell[0], "screening_min_reliability": cell[1],
                         extra_col: i, "spearman": v})
    return pd.DataFrame(rows)


class TestPower:
    def _null_and_alt(self, shift):
        rng = np.random.default_rng(0)
        null = rng.normal(0.50, 0.01, 200)
        alt = null - shift + rng.normal(0, 0.01, 200)
        return (_draws([(50, 0.0)], [null]), _draws([(50, 0.0)], [alt]))

    def test_no_effect_gives_power_near_alpha(self):
        null, alt = self._null_and_alt(0.0)
        out = rp.power_table(null, alt, target_rho=1.0)
        assert out["power"].iloc[0] < 0.25

    def test_a_large_effect_gives_power_near_one(self):
        null, alt = self._null_and_alt(0.10)
        out = rp.power_table(null, alt, target_rho=0.80)
        assert out["power"].iloc[0] > 0.95

    def test_it_reports_the_two_correlations_behind_the_number(self):
        """Power alone is uninterpretable without the ceiling and the observed value."""
        null, alt = self._null_and_alt(0.05)
        row = rp.power_table(null, alt, target_rho=0.90).iloc[0]
        assert row["ceiling"] == pytest.approx(0.50, abs=0.01)
        assert row["observed"] == pytest.approx(0.45, abs=0.02)
        assert row["drop_below_ceiling"] == pytest.approx(0.05, abs=0.02)

    def test_the_critical_value_is_per_cell(self):
        """The ceiling and its spread both depend on N, so one global cut would be wrong."""
        rng = np.random.default_rng(1)
        null = _draws([(50, 0.0), (75, 0.0)],
                      [rng.normal(0.44, 0.02, 100), rng.normal(0.52, 0.01, 100)])
        alt = _draws([(50, 0.0), (75, 0.0)],
                     [rng.normal(0.43, 0.02, 100), rng.normal(0.51, 0.01, 100)])
        out = rp.power_table(null, alt, target_rho=0.98)
        assert out["critical_value"].nunique() == 2
        assert out.set_index("num_subjects").loc[75, "critical_value"] > \
               out.set_index("num_subjects").loc[50, "critical_value"]

    def test_missing_cell_columns_raise(self):
        bad = pd.DataFrame({"spearman": [0.5]})
        with pytest.raises(ValueError, match="cell columns"):
            rp.power_table(bad, bad, target_rho=0.9)


class TestMinimumDetectableEffect:
    def _curve(self, powers):
        return pd.DataFrame([{"num_subjects": 50, "screening_min_reliability": 0.0,
                              "target_rho": r, "power": p}
                             for r, p in powers])

    def test_interpolates_between_simulated_points(self):
        curve = self._curve([(0.99, 0.20), (0.98, 0.60), (0.95, 0.95)])
        mde = rp.minimum_detectable_effect(curve)["min_detectable_rho"].iloc[0]
        assert 0.95 < mde < 0.98

    def test_returns_nan_when_the_target_is_never_reached(self):
        """Extrapolating would read as a measurement; 'not within the simulated range' is honest."""
        curve = self._curve([(0.99, 0.10), (0.98, 0.20), (0.95, 0.40)])
        assert np.isnan(rp.minimum_detectable_effect(curve)["min_detectable_rho"].iloc[0])

    def test_returns_the_smallest_effect_when_everything_is_detectable(self):
        curve = self._curve([(0.99, 0.90), (0.98, 0.95), (0.95, 0.99)])
        assert rp.minimum_detectable_effect(curve)["min_detectable_rho"].iloc[0] == 0.99

    def test_effect_percentage_matches_the_rho(self):
        curve = self._curve([(0.99, 0.20), (0.98, 0.60), (0.95, 0.95)])
        row = rp.minimum_detectable_effect(curve).iloc[0]
        assert row["min_detectable_effect_pct"] == pytest.approx(100 * (1 - row["min_detectable_rho"]))
