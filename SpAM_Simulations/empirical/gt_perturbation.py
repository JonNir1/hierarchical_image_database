"""A second ground truth, a controlled distance apart from the first.

RQ2 asks whether SHINE preserves perceptual structure. Simulating its power needs a *pair* of
ground truths whose true similarity is known: a "pre-SHINE" space and a "post-SHINE" space that
differ by exactly ``rho_true``. Everything else in this package compares a cohort against one fixed
ground truth, so this is new.

**Isotropic by construction, and that is a real limitation.** The perturbation is added to every
coordinate equally, so it makes no claim about *where* SHINE acts. If SHINE instead acts selectively
on sensory dimensions - which is what RQ3 supposes - the power this yields is optimistic or
pessimistic depending on how much of the distance variance those dimensions carry.
``analysis/rdms/`` already holds ``sens_pre``/``sens_post``, so a sensory-weighted variant is the
natural follow-up; it is deliberately not attempted here, because deriving the sensory/semantic
split of the ground truth's axes would make the power estimate conditional on that derivation.

The target is stated on the **condensed-distance Spearman**, not on the coordinates, because that
is the quantity RQ2's test statistic is computed on. Two embeddings can differ substantially in
coordinates while inducing near-identical distances, so a coordinate-space perturbation magnitude
would not be interpretable as an effect size.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

DEFAULT_TOLERANCE = 0.002
DEFAULT_MAX_ITERS = 60

# The bracket search doubles until the similarity drops below the target. For an ordinary
# perturbation direction that happens by ~5x the coordinate spread (rho is already <0.05 there), so
# a cap of 50x is generous. It exists because a DEGENERATE direction never triggers the stop at all:
# if the noise happens to be parallel to the coordinates, `coords + scale * noise` is a pure
# rescale, every distance grows by the same factor, and the Spearman stays at exactly 1.0 no matter
# how large the scale gets. Without the cap the loop doubles to ~1e17 and returns nonsense.
MAX_BRACKET_SPREADS = 50.0


def distance_similarity(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Spearman between the two embeddings' condensed distance vectors."""
    rho = spearmanr(pdist(np.asarray(coords_a, dtype=np.float64)),
                    pdist(np.asarray(coords_b, dtype=np.float64)))[0]
    return float(rho)


def perturb_gt(coords: np.ndarray, target_rho: float, *, seed: int = 0,
               tolerance: float = DEFAULT_TOLERANCE,
               max_iters: int = DEFAULT_MAX_ITERS) -> Tuple[np.ndarray, Dict[str, float]]:
    """A copy of ``coords`` whose distances correlate with the original at ``target_rho``.

    Adds isotropic Gaussian noise to the coordinates and bisects its scale until the induced
    condensed-distance Spearman lands within ``tolerance`` of the target. Bisection rather than a
    closed form because the map from coordinate noise to distance correlation has no convenient
    inverse and depends on the embedding's own spectrum.

    The noise is drawn ONCE and only its scale is searched, so the returned embedding is a
    continuous function of the scale and the bisection is monotone. Redrawing inside the loop would
    make the objective jump around and the search would not converge.

    Returns ``(perturbed_coords, info)``, where ``info`` records the achieved rho, the scale that
    produced it, and whether the search converged. Two distinct failure modes, deliberately handled
    differently. A direction that cannot bracket the target at all **raises** - that means the
    perturbation is degenerate and no answer exists. A direction that brackets but does not refine
    to within ``tolerance`` in ``max_iters`` returns its closest attempt with ``converged=False``,
    since that value is still usable and the caller can decide. **Check ``converged``**: a silently
    missed target would misstate the effect size the power curve is plotted against.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if not 0.0 < target_rho <= 1.0:
        raise ValueError(f"target_rho must be in (0, 1], got {target_rho}")
    if target_rho == 1.0:
        return coords.copy(), {"target_rho": 1.0, "achieved_rho": 1.0, "noise_scale": 0.0,
                               "iters": 0, "converged": True}

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(coords.shape)
    spread = float(coords.std())

    # Bracket first: grow the upper bound until it overshoots, so bisection has something to work
    # with. A fixed bracket would silently fail for an embedding with an unusual spectrum.
    cap = MAX_BRACKET_SPREADS * spread
    lo, hi = 0.0, 0.25 * spread
    while hi <= cap:
        if distance_similarity(coords, coords + hi * noise) <= target_rho:
            break
        lo, hi = hi, hi * 2.0
    else:
        achieved = distance_similarity(coords, coords + hi * noise)
        raise ValueError(
            f"could not bracket target_rho={target_rho}: the similarity is still {achieved:.4f} at "
            f"{MAX_BRACKET_SPREADS:g}x the coordinate spread. The perturbation direction is "
            f"degenerate - if the noise is parallel to the coordinates the perturbation is a pure "
            f"rescale, which leaves every distance rank unchanged. Try a different seed.")

    achieved, scale = np.nan, hi
    for i in range(max_iters):
        scale = 0.5 * (lo + hi)
        achieved = distance_similarity(coords, coords + scale * noise)
        if abs(achieved - target_rho) <= tolerance:
            return coords + scale * noise, {
                "target_rho": target_rho, "achieved_rho": achieved, "noise_scale": scale,
                "iters": i + 1, "converged": True}
        # Similarity falls monotonically with the noise scale.
        lo, hi = (scale, hi) if achieved > target_rho else (lo, scale)

    return coords + scale * noise, {"target_rho": target_rho, "achieved_rho": achieved,
                                    "noise_scale": scale, "iters": max_iters,
                                    "converged": bool(abs(achieved - target_rho) <= tolerance)}


def build_perturbed_set(coords: np.ndarray, targets, *, seed: int = 0,
                        **kwargs) -> Dict[float, Tuple[np.ndarray, Dict[str, float]]]:
    """``{target_rho: (coords, info)}`` for several targets, each with its own seed.

    Distinct seeds per target so the perturbations are independent draws rather than one direction
    scaled up and down; sharing a direction would make the power curve smoother than it should be
    and understate how much the answer depends on where the perturbation lands.
    """
    return {float(t): perturb_gt(coords, float(t), seed=seed + i, **kwargs)
            for i, t in enumerate(targets)}
