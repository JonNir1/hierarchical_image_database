"""Balanced incomplete block designs for allocating images to SpAM trials.

The deployed task assigns images to trials by a per-participant shuffle sliced into consecutive
trials (``SpAM_Task/js/trial_generator.js::buildTrialLists``), independently per subject and with
no coordination across the cohort. That is cheap but wasteful: pairs are discovered rather than
replicated, so coverage broadens instead of deepening and per-image replication varies widely.

A balanced design instead chooses trials so that every *pair* of images co-occurs about equally
often. This module implements MacDonald's "best of greedy" construction
(https://doi.org/10.3758/s13428-019-01326-x, reference code at
https://justinmacdonald.net/static/best_of_greedy_algorithm.py) in vectorised form, plus a
session-aware variant the simulations actually need.

Terminology maps onto the task as: *treatment* = image, *block* = trial, *lambda* = the number of
times each image pair should co-occur.

**Vectorisation.** The reference implementation is O(n*k^2 + n^2*k) per block with Python loops and
does not finish at n=725. The pair-deficit matrix is symmetric and stays symmetric (decrements are
applied to both ``[a,b]`` and ``[b,a]``), so the reference's per-candidate ``links`` reduces to
``2 * sum_j deficit[s, chosen_j]`` and can be accumulated incrementally as each image is picked.
``degree`` is likewise maintained rather than recomputed. That makes it O(n*k) per block: ~1.7 s
for a full n=725, k=20, lambda=1 design instead of hours. Tie-breaking follows the reference
exactly (max links, then max degree among those, then uniform random), so the search behaviour is
unchanged.

**Why a session-aware variant is required.** A global design cannot be partitioned into subject
sessions after the fact. At n=725, k=20, lambda=1 the design has ~2157 blocks and each image sits
in ~59.5 of them, so a given block shares an image with roughly 20*58.5 ~ 1170 of the other 2156
blocks - a conflict density near 54%. Drawing 18 mutually image-disjoint blocks from that has
probability on the order of 0.46^153. :func:`greedy_session_design` therefore imposes the
constraint *during* generation via a per-session used-image mask, which preserves the "each image
appears at most once per subject" guarantee that every existing simulation relies on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def schonheim(n_images: int, k: int) -> int:
    """Schonheim lower bound on the number of blocks needed to cover every pair once.

    ``C(v, k, 2) >= ceil(v/k * ceil((v-1)/(k-1)))``. A greedy construction cannot beat this, so it
    is the yardstick for how much a design is leaving on the table (at n=725, k=20 it is 1414).
    """
    return -(-n_images * (-(-(n_images - 1) // (k - 1))) // k)


def _pick(links: np.ndarray, degree: np.ndarray, chosen: np.ndarray, n_chosen: int,
          rng: np.random.Generator, banned: Optional[np.ndarray] = None) -> int:
    """One greedy pick: max links, ties broken by max degree, then uniformly at random."""
    cand = links.copy()
    if n_chosen:
        cand[chosen[:n_chosen]] = -1
    if banned is not None:
        cand[banned] = -1
    idx = np.flatnonzero(cand == cand.max())
    if idx.size == 1:
        return int(idx[0])
    dsub = degree[idx]
    idx = idx[dsub == dsub.max()]
    return int(idx[0] if idx.size == 1 else idx[rng.integers(idx.size)])


class _PairDeficit:
    """The symmetric ``lambda - (times the pair has been covered)`` matrix, floored at zero."""

    def __init__(self, n_images: int, lam: int) -> None:
        self.m = np.full((n_images, n_images), lam, dtype=np.int32)
        np.fill_diagonal(self.m, 0)
        self.degree = self.m.sum(axis=0)
        self.remaining = int(self.m.sum())

    def cover(self, new: int, prev: np.ndarray) -> None:
        """Record that ``new`` co-occurred with each already-chosen image in ``prev``."""
        if prev.size == 0:
            return
        hit = prev[self.m[new, prev] > 0]     # floor at 0, as in the reference implementation
        if hit.size == 0:
            return
        self.m[new, hit] -= 1
        self.m[hit, new] -= 1
        self.degree[new] -= hit.size
        self.degree[hit] -= 1
        self.remaining -= 2 * hit.size


def greedy_design(n_images: int, k: int, lam: int, rng: np.random.Generator) -> np.ndarray:
    """One greedy covering design. Returns ``(n_blocks, k)`` of 0-based image indices.

    Blocks are emitted until every pair has co-occurred at least ``lam`` times, so the result is a
    valid covering (``min`` pair count ``>= lam``) but not generally a balanced incomplete block
    design - an exact BIBD needs ``lam*(v-1) % (k-1) == 0``, which n=725, k=20 fails for every
    lambda below 19.
    """
    deficit = _PairDeficit(n_images, lam)
    blocks: List[np.ndarray] = []
    while deficit.remaining > 0:
        block = np.empty(k, dtype=np.int64)
        links = np.zeros(n_images, dtype=np.int64)
        for i in range(k):
            block[i] = pick = _pick(links, deficit.degree, block, i, rng)
            links += 2 * deficit.m[:, pick]
            deficit.cover(pick, block[:i])
        blocks.append(block)
    return np.asarray(blocks, dtype=np.int32)


def best_of_greedy(n_images: int, k: int, lam: int, m: int, rng: np.random.Generator,
                   ) -> Tuple[np.ndarray, List[int]]:
    """Run :func:`greedy_design` ``m`` times and keep the smallest. Returns ``(design, sizes)``.

    MacDonald's "best of" step. It matters at small ``n``, where greedy outcomes vary; at n=725 the
    spread is ~0.3% across runs, so ``m=1`` is usually the right call there (see the module tests).
    """
    best: Optional[np.ndarray] = None
    sizes: List[int] = []
    for _ in range(m):
        d = greedy_design(n_images, k, lam, rng)
        sizes.append(int(d.shape[0]))
        if best is None or d.shape[0] < best.shape[0]:
            best = d
    assert best is not None, "m must be >= 1"
    return best, sizes


def greedy_session_design(n_images: int, k: int, trials_per_session: int, n_sessions: int,
                          rng: np.random.Generator, lam: int = 1, m: int = 1) -> np.ndarray:
    """A balanced design partitioned into per-subject sessions of image-disjoint trials.

    Returns ``(n_sessions, trials_per_session, k)`` of 0-based image indices. Within a session every
    image appears at most once, matching what the deployed task guarantees, so a session is a drop-in
    replacement for one subject's trial list.

    The global pair deficit is shared across sessions and drives the greedy choice, so sessions
    jointly cover the pair space rather than each covering it independently. When the deficit is
    exhausted it is refilled to ``lam``, letting the generator keep producing balanced sessions past
    one full covering (needed whenever ``n_sessions * trials_per_session`` exceeds the covering size).

    ``m`` is the "best of" count. Note it cannot mean what it means in :func:`best_of_greedy`: the
    block count here is fixed at ``n_sessions * trials_per_session``, so there is no size to
    minimise. Instead ``m`` whole designs are generated and the one covering the most distinct pairs
    is kept (ties broken by the flatter per-image replication).
    """
    if trials_per_session * k > n_images:
        raise ValueError(
            f"a session needs {trials_per_session * k} distinct images "
            f"(trials_per_session={trials_per_session} x k={k}) but only {n_images} exist"
        )
    if m > 1:
        best, best_key = None, None
        for _ in range(m):
            cand = greedy_session_design(n_images, k, trials_per_session, n_sessions, rng, lam, m=1)
            stats = session_design_stats(cand, n_images)
            key = (stats["frac_pairs_covered"], -stats["reps_per_image_sd"])
            if best_key is None or key > best_key:
                best, best_key = cand, key
        assert best is not None, "m must be >= 1"
        return best
    deficit = _PairDeficit(n_images, lam)
    sessions = np.empty((n_sessions, trials_per_session, k), dtype=np.int32)
    for s in range(n_sessions):
        used = np.zeros(n_images, dtype=bool)
        for t in range(trials_per_session):
            block = np.empty(k, dtype=np.int64)
            links = np.zeros(n_images, dtype=np.int64)
            banned = np.flatnonzero(used)
            for i in range(k):
                block[i] = pick = _pick(links, deficit.degree, block, i, rng, banned=banned)
                links += 2 * deficit.m[:, pick]
                deficit.cover(pick, block[:i])
            used[block] = True
            sessions[s, t] = block
        if deficit.remaining == 0:
            deficit = _PairDeficit(n_images, lam)
    return sessions


def session_design_stats(sessions: np.ndarray, n_images: int) -> Dict[str, float]:
    """Coverage/balance summary for a ``(n_sessions, t, k)`` design, for the Stage 2a comparison."""
    flat = sessions.reshape(-1, sessions.shape[-1])
    return {**design_stats(flat, n_images),
            "n_sessions": int(sessions.shape[0]),
            "trials_per_session": int(sessions.shape[1])}


def design_stats(blocks: np.ndarray, n_images: int) -> Dict[str, float]:
    """Coverage/balance summary for a flat ``(n_blocks, k)`` design."""
    counts = pair_counts(blocks, n_images)
    reps = np.bincount(blocks.ravel(), minlength=n_images)
    total = int(counts.sum())
    return {
        "n_blocks": int(blocks.shape[0]),
        "n_pairs": int(counts.size),
        "frac_pairs_covered": float((counts > 0).mean()),
        "min_pair_count": int(counts.min()),
        "mean_pair_count": float(counts.mean()),
        "max_pair_count": int(counts.max()),
        "reps_per_image_mean": float(reps.mean()),
        "reps_per_image_min": int(reps.min()),
        "reps_per_image_max": int(reps.max()),
        "reps_per_image_sd": float(reps.std()),
        # ratings an image actually has, out of the n_images-1 partners it could have
        "partners_per_image_mean": float(_partner_counts(counts, n_images).mean()),
        "wasted_frac": float((counts - 1).clip(0).sum() / total) if total else 0.0,
    }


def pair_counts(blocks: np.ndarray, n_images: int) -> np.ndarray:
    """Condensed per-pair co-occurrence counts for a ``(n_blocks, k)`` design."""
    square = np.zeros((n_images, n_images), dtype=np.int32)
    for b in blocks:
        square[np.ix_(b, b)] += 1
    np.fill_diagonal(square, 0)
    return square[np.triu_indices(n_images, 1)]


def _partner_counts(condensed: np.ndarray, n_images: int) -> np.ndarray:
    """Per-image count of distinct partners observed at least once."""
    square = np.zeros((n_images, n_images), dtype=np.int32)
    square[np.triu_indices(n_images, 1)] = condensed
    square += square.T
    return (square > 0).sum(axis=1)
