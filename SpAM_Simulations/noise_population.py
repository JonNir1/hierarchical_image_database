"""The population of per-subject placement precisions, and its shape.

Every task-vN model gives subject ``s`` a placement-precision scalar ``sigma_s`` drawn once and
reused for all their trials. Until now that draw was fixed to ``|t(df)|`` rescaled to a target mean
(``experiment._draw_subject_noises``), with only the *mean* ever calibrated to the pilot - the
**shape** was assumed.

Checking the simulation against 36 subjects with usable whole-trial repeats (22 at task v3.\\*, 14 at
v4.0) showed the assumed shape is wrong at both tails: at ``df=5`` the simulated cohort contains far
more subjects with a catastrophic repeat than reality does, and also more excellent ones. Real
subjects are more **concentrated**. Since screening's entire yield is set by how dispersed this
population is - it can only truncate a distribution, never shift it - getting the shape right is a
precondition for any trustworthy statement about screening or required-N.

**Why ``df`` alone cannot fix it.** ``|t(df)|`` normalised to a fixed mean has a *floor* on its
coefficient of variation: as ``df -> inf`` it converges to a half-normal, whose CV is
``sqrt(pi/2 - 1) ~ 0.756``. Going from ``df=5`` (CV 0.919) to ``df=50`` (CV 0.764) buys only ~17%
less dispersion and nothing beyond that, so the whole family is confined to CV in
``[0.756, inf)``. A population more concentrated than half-normal is simply not expressible.

This module therefore offers two families behind one interface:

* ``"t"``          - ``|t(df)|``, the historical behaviour. ``shape`` is ``df``. CV floor 0.756.
* ``"lognormal"``  - ``exp(N(0, s))``, ``shape`` is ``s``. CV is ``sqrt(exp(s^2) - 1)``, which spans
  ``(0, inf)`` continuously: ``s -> 0`` is a homogeneous cohort (everyone identical), ``s = 0.7``
  reproduces roughly the half-normal's dispersion, and large ``s`` is heavier-tailed than any ``t``.

Both are rescaled so the *sample* mean equals ``mean_scale``, matching the original estimator, so
``mean_scale`` keeps its meaning (and its calibration) across families.
"""
from __future__ import annotations

import numpy as np

FAMILIES = ("t", "lognormal")

# `subjects_noise_lognormal_sigma` uses 0.0 as "not lognormal, use the t family with
# subjects_noise_df" - a sentinel rather than a separate categorical field, because the experiment
# parameter tuples must stay entirely numeric (the MDS pipeline coerces every parameter to float).
LOGNORMAL_DISABLED = 0.0


def draw_subject_noises(
        n_subjects: int,
        mean_scale: float,
        *,
        rng: np.random.Generator,
        family: str = "t",
        shape: float = 5.0,
) -> np.ndarray:
    """Draw ``n_subjects`` placement precisions with sample mean exactly ``mean_scale``.

    :param family: ``"t"`` (``shape`` = degrees of freedom) or ``"lognormal"`` (``shape`` = sigma).
    :param shape: the dispersion parameter of the chosen family; must be positive.

    Rescaling by the sample mean (rather than the population mean) is deliberate: it is what
    ``experiment._draw_subject_noises`` has always done, so ``mean_scale`` remains comparable with
    every previous calibration. It also means a *selected* subsample - e.g. the survivors of
    screening - is free to have a lower mean, which is exactly the effect under study.
    """
    if n_subjects <= 0:
        raise ValueError(f"`n_subjects` must be positive, got {n_subjects}")
    if mean_scale < 0:
        raise ValueError(f"`mean_scale` must be non-negative, got {mean_scale}")
    if shape <= 0:
        raise ValueError(f"`shape` must be positive, got {shape}")
    if family not in FAMILIES:
        raise ValueError(f"`family` must be one of {FAMILIES}, got {family!r}")
    if mean_scale == 0:
        return np.zeros(n_subjects)
    raw = (np.abs(rng.standard_t(shape, size=n_subjects)) if family == "t"
           else np.exp(rng.normal(0.0, shape, size=n_subjects)))
    return mean_scale * raw / np.mean(raw)


def population_cv(family: str, shape: float, *, n: int = 200_000, seed: int = 0) -> float:
    """Coefficient of variation of a family/shape, by simulation. Diagnostic for fit reports.

    The CV is the scale-free summary of how heterogeneous a cohort is, and it is what bounds
    screening's yield - hence reporting it alongside any fitted population.
    """
    x = draw_subject_noises(n, 1.0, rng=np.random.default_rng(seed), family=family, shape=shape)
    return float(np.std(x) / np.mean(x))


def resolve_family(noise_df: float, lognormal_sigma: float) -> tuple:
    """Map the two numeric experiment parameters onto ``(family, shape)``.

    ``lognormal_sigma > 0`` selects the lognormal family and is its sigma; ``0.0`` (the sentinel
    :data:`LOGNORMAL_DISABLED`) falls back to ``|t(noise_df)|``, i.e. the historical behaviour, so a
    parameter tuple written before this module existed keeps its exact meaning.
    """
    if lognormal_sigma and lognormal_sigma > 0:
        return "lognormal", float(lognormal_sigma)
    return "t", float(noise_df)
