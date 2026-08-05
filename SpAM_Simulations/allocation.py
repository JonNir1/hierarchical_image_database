"""Strategies for allocating images to a subject's trials.

The deployed task, and therefore every simulation to date, allocates randomly and independently
per subject: ``seededShuffle(allImages).slice(0, n_needed)`` chopped into consecutive trials
(``SpAM_Task/js/trial_generator.js:56``), mirrored by ``rng.choice`` in
``task_v4_experiment.simulate_task_v4_experiment``. This module makes that choice explicit so a
balanced alternative can be swept against it as an experimental arm rather than a code fork.

Two allocators, both returning a :class:`SubjectAllocation` of explicit trial lists:

* :class:`RandomAllocator` reproduces the current behaviour. It exists for symmetry in the sweep;
  the simulation keeps its own inline path for ``allocation_mode == RANDOM`` so the default arm
  stays bit-for-bit identical to the published runs (see ``test_bit_exact_v4.py``).
* :class:`DesignedAllocator` hands out pre-generated sessions from
  ``block_design.greedy_session_design``, where trials are chosen so image pairs co-occur about
  equally often across the cohort.

**Screened-out candidates do not burn a design slot.** ``DesignedAllocator.draw`` is paired with
``commit``/``rollback``: a candidate who fails screening returns their session to the pool. The
design is a plan over the subjects who end up *analysed*, so consuming a slot per candidate would
silently degrade coverage exactly in proportion to the screening rejection rate, confounding the
arm comparison with the screening threshold.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional

import numpy as np

# Numeric because the value travels inside TaskV4ExperimentParameters, which pipeline._task_key
# coerces field-by-field with float() and _completed_keys rebuilds from meta.csv.
RANDOM = 0.0
DESIGNED = 1.0

SubjectAllocation = NamedTuple("SubjectAllocation", [
    ("screen", Optional[List[np.ndarray]]),   # screening-stage trials, or None when not screening
    ("main", List[np.ndarray]),               # main-stage trials
])


class RandomAllocator:
    """Draws one disjoint pool per subject and slices it into trials, as the deployed task does."""

    def __init__(self, n_images: int, k: int, screen_distinct: int, main_distinct: int) -> None:
        self.n_images = n_images
        self.k = k
        self.screen_distinct = screen_distinct
        self.main_distinct = main_distinct

    def draw(self, rng: np.random.Generator) -> SubjectAllocation:
        need = (self.screen_distinct + self.main_distinct) * self.k
        pool = rng.choice(self.n_images, size=need, replace=False)
        trials = [pool[i * self.k:(i + 1) * self.k]
                  for i in range(self.screen_distinct + self.main_distinct)]
        screen = trials[:self.screen_distinct] if self.screen_distinct else None
        return SubjectAllocation(screen=screen, main=trials[self.screen_distinct:])

    def commit(self) -> None:      # nothing to consume
        pass

    def rollback(self) -> None:    # nothing to return
        pass


class DesignedAllocator:
    """Hands out pre-generated balanced sessions, one per retained subject.

    ``sessions`` is ``(n_sessions, trials_per_session, k)`` from
    ``block_design.greedy_session_design``. The first ``screen_distinct`` trials of a session go to
    the screening stage and the rest to the main stage, matching
    ``trial_generator.js::partitionIntoStages``; because a session's trials are image-disjoint by
    construction, no image can appear in both stages.
    """

    def __init__(self, sessions: np.ndarray, screen_distinct: int) -> None:
        sessions = np.asarray(sessions)
        if sessions.ndim != 3:
            raise ValueError(f"`sessions` must be (n_sessions, trials, k), got {sessions.shape}")
        if screen_distinct >= sessions.shape[1]:
            raise ValueError(
                f"screen_distinct={screen_distinct} leaves no main-stage trials in a session of "
                f"{sessions.shape[1]}"
            )
        self.sessions = sessions
        self.screen_distinct = screen_distinct
        self._next = 0
        self._pending: Optional[int] = None

    @property
    def n_sessions(self) -> int:
        return int(self.sessions.shape[0])

    def draw(self, rng: np.random.Generator) -> SubjectAllocation:
        """Take the next unconsumed session. ``rng`` is unused; the design is deterministic."""
        if self._next >= self.n_sessions:
            raise RuntimeError(
                f"designed allocation exhausted after {self.n_sessions} sessions; generate a design "
                "with more sessions than the expected number of screened candidates"
            )
        self._pending = self._next
        self._next += 1
        session = self.sessions[self._pending]
        trials = [np.asarray(t) for t in session]
        screen = trials[:self.screen_distinct] if self.screen_distinct else None
        return SubjectAllocation(screen=screen, main=trials[self.screen_distinct:])

    def commit(self) -> None:
        """Keep the drawn session: this subject was retained."""
        self._pending = None

    def rollback(self) -> None:
        """Return the drawn session to the pool: this candidate was screened out."""
        if self._pending is not None:
            self._next = self._pending
            self._pending = None


def make_allocator(allocation_mode: float, *, n_images: int, k: int, screen_distinct: int,
                   main_distinct: int, sessions: Optional[np.ndarray] = None):
    """Build the allocator a sweep cell's numeric ``allocation_mode`` asks for."""
    if float(allocation_mode) == DESIGNED:
        if sessions is None:
            raise ValueError("allocation_mode=DESIGNED requires `sessions`")
        return DesignedAllocator(sessions, screen_distinct)
    return RandomAllocator(n_images, k, screen_distinct, main_distinct)
