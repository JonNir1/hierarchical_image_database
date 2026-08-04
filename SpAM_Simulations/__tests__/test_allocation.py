"""Tests for image-to-trial allocation strategies and the allocation_mode seam.

The bit-exactness of the default (random) arm is covered by test_bit_exact_v4.py; these tests
cover the designed arm's own guarantees and the wiring that lets a sweep carry both.

No R needed.
"""
import numpy as np
import pytest

from SpAM_Simulations.allocation import (
    DESIGNED, RANDOM, DesignedAllocator, RandomAllocator, make_allocator,
)
from SpAM_Simulations.block_design import greedy_session_design, pair_counts
from SpAM_Simulations.config import TaskV4SimulationConfig
from SpAM_Simulations.simulation import build_ground_truth_embeddings
from SpAM_Simulations.task_v4_experiment import (
    TaskV4ExperimentParameters, simulate_task_v4_experiment,
)

N_IMAGES, K = 40, 5
GT = build_ground_truth_embeddings(N_IMAGES, 4, seed=42)


def _sessions(n_sessions=12, trials=4):
    return greedy_session_design(N_IMAGES, K, trials, n_sessions, np.random.default_rng(0))


def _params(**overrides):
    fields = dict(
        num_subjects=3, trials_per_subject=4, images_per_trial=K,
        subjects_noise_scale=0.4, subjects_noise_df=5,
        frac_trials_repeated=0.25, perspective_dispersion=0.2,
        screening_trials=0, screening_repeats=0, screening_min_reliability=-1.0,
        subjects_noise_lognormal_sigma=0.0, allocation_mode=RANDOM,
    )
    fields.update(overrides)
    return TaskV4ExperimentParameters(**fields)


# --------------------------------------------------------------------- allocators

def test_random_allocator_draws_a_disjoint_pool():
    alloc = RandomAllocator(N_IMAGES, K, screen_distinct=1, main_distinct=3)
    a = alloc.draw(np.random.default_rng(0))
    flat = np.concatenate([*a.screen, *a.main])
    assert flat.size == 4 * K
    assert np.unique(flat).size == flat.size, "a subject must not see an image twice"


def test_designed_allocator_hands_out_sessions_in_order():
    sessions = _sessions()
    alloc = DesignedAllocator(sessions, screen_distinct=0)
    for i in range(3):
        drawn = np.concatenate(alloc.draw(np.random.default_rng(0)).main)
        np.testing.assert_array_equal(drawn, sessions[i].ravel())
        alloc.commit()


def test_rollback_returns_the_session_to_the_pool():
    """A screened-out candidate must not consume a design slot."""
    sessions = _sessions()
    alloc = DesignedAllocator(sessions, screen_distinct=0)
    first = alloc.draw(np.random.default_rng(0))
    alloc.rollback()
    again = alloc.draw(np.random.default_rng(0))
    np.testing.assert_array_equal(np.concatenate(first.main), np.concatenate(again.main))


def test_commit_consumes_the_session():
    alloc = DesignedAllocator(_sessions(), screen_distinct=0)
    first = alloc.draw(np.random.default_rng(0))
    alloc.commit()
    second = alloc.draw(np.random.default_rng(0))
    assert not np.array_equal(np.concatenate(first.main), np.concatenate(second.main))


def test_exhausted_designed_allocation_raises():
    alloc = DesignedAllocator(_sessions(n_sessions=2), screen_distinct=0)
    for _ in range(2):
        alloc.draw(np.random.default_rng(0))
        alloc.commit()
    with pytest.raises(RuntimeError, match="exhausted"):
        alloc.draw(np.random.default_rng(0))


def test_designed_allocator_splits_stages_without_sharing_images():
    alloc = DesignedAllocator(_sessions(trials=4), screen_distinct=1)
    a = alloc.draw(np.random.default_rng(0))
    assert len(a.screen) == 1 and len(a.main) == 3
    assert not (set(np.concatenate(a.screen).tolist()) & set(np.concatenate(a.main).tolist()))


def test_designed_allocator_rejects_a_screening_split_leaving_no_main_trials():
    with pytest.raises(ValueError, match="no main-stage trials"):
        DesignedAllocator(_sessions(trials=4), screen_distinct=4)


def test_make_allocator_dispatches_on_mode():
    kw = dict(n_images=N_IMAGES, k=K, screen_distinct=0, main_distinct=4)
    assert isinstance(make_allocator(RANDOM, **kw), RandomAllocator)
    assert isinstance(make_allocator(DESIGNED, sessions=_sessions(), **kw), DesignedAllocator)
    with pytest.raises(ValueError, match="requires `sessions`"):
        make_allocator(DESIGNED, **kw)


# --------------------------------------------------------------------- the seam

def test_designed_arm_observes_exactly_the_designed_pairs():
    """The design must survive intact: build_trial_lists would have reshuffled it away."""
    sessions = _sessions(n_sessions=3, trials=3)
    params = _params(num_subjects=3, trials_per_subject=3, frac_trials_repeated=0.0,
                     allocation_mode=DESIGNED)
    alloc = DesignedAllocator(sessions, screen_distinct=0)
    _, res = simulate_task_v4_experiment(params, GT, np.random.default_rng(0),
                                         verbose=False, allocator=alloc)
    expected = pair_counts(sessions.reshape(-1, K), N_IMAGES)
    np.testing.assert_array_equal(res.num_obs.astype(bool), expected.astype(bool))


def test_allocator_none_is_the_untouched_default_path():
    """Belt and braces alongside test_bit_exact_v4: explicit None must equal omitting it."""
    p = _params()
    _, a = simulate_task_v4_experiment(p, GT, np.random.default_rng(7), verbose=False)
    _, b = simulate_task_v4_experiment(p, GT, np.random.default_rng(7), verbose=False,
                                       allocator=None)
    np.testing.assert_array_equal(np.nan_to_num(a.distances, nan=-1),
                                  np.nan_to_num(b.distances, nan=-1))


def test_screened_out_candidates_do_not_burn_design_slots():
    """Sessions consumed must equal subjects RETAINED, not candidates screened.

    Otherwise the designed arm's coverage would silently degrade in proportion to the rejection
    rate, confounding the arm comparison with the screening threshold.

    The threshold rejects some candidates but not all, which is the case that actually exercises
    rollback. A threshold nothing can satisfy now raises instead (see
    test_task_v4_experiment.TestRecruitmentCap), so it would not reach the assertion below.
    """
    sessions = _sessions(n_sessions=30, trials=4)
    alloc = DesignedAllocator(sessions, screen_distinct=1)
    params = _params(screening_trials=2, screening_repeats=1, screening_min_reliability=0.3,
                     allocation_mode=DESIGNED, num_subjects=3, subjects_noise_scale=0.8)
    _, res = simulate_task_v4_experiment(params, GT, np.random.default_rng(0), verbose=False,
                                         allocator=alloc)
    assert res.n_candidates_screened > params.num_subjects, "threshold rejected nobody; pick a stricter one"
    assert alloc._next == params.num_subjects, (
        f"consumed {alloc._next} sessions for {params.num_subjects} retained subjects "
        f"out of {res.n_candidates_screened} candidates"
    )


def test_trials_and_image_indices_are_mutually_exclusive():
    from SpAM_Simulations.task_v4_experiment import simulate_task_v4_single_subject
    with pytest.raises(AssertionError, match="not both"):
        simulate_task_v4_single_subject(
            subject_noise=0.3, perspective_dispersion=0.2, t_distinct=2, k=K, n_unique=2 * K,
            n_repeats=0, gt_embeddings=GT, rng=np.random.default_rng(0),
            image_indices=np.arange(2 * K), trials=[np.arange(K), np.arange(K, 2 * K)],
        )


# --------------------------------------------------------------------- config lever

def test_config_sweeps_allocation_mode_as_a_lever():
    cfg = TaskV4SimulationConfig(
        gt_embeddings=GT, num_subjects=[3], trials_per_subject=[4], images_per_trial=[K],
        subjects_noise_scale=[0.4], subjects_noise_df=[5], frac_trials_repeated=[0.25],
        perspective_dispersion=[0.2], screening_trials=[0], screening_repeats=[0],
        screening_min_reliability=[-1.0], allocation_mode=[RANDOM, DESIGNED],
    )
    modes = sorted({p.allocation_mode for p in cfg.param_grid()})
    assert modes == [RANDOM, DESIGNED]


def test_config_defaults_to_the_random_arm():
    cfg = TaskV4SimulationConfig(
        gt_embeddings=GT, num_subjects=[3], trials_per_subject=[4], images_per_trial=[K],
        subjects_noise_scale=[0.4], subjects_noise_df=[5], frac_trials_repeated=[0.25],
        perspective_dispersion=[0.2], screening_trials=[0], screening_repeats=[0],
        screening_min_reliability=[-1.0],
    )
    assert {p.allocation_mode for p in cfg.param_grid()} == {RANDOM}


def test_config_rejects_an_unknown_allocation_mode():
    with pytest.raises(ValueError, match="allocation_mode"):
        TaskV4SimulationConfig(
            gt_embeddings=GT, num_subjects=[3], trials_per_subject=[4], images_per_trial=[K],
            subjects_noise_scale=[0.4], subjects_noise_df=[5], frac_trials_repeated=[0.25],
            perspective_dispersion=[0.2], screening_trials=[0], screening_repeats=[0],
            screening_min_reliability=[-1.0], allocation_mode=[0.5],
        )
