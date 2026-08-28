"""Scheduling is pure Python -- these run without torch or any weights."""

import pytest

from vsr.scheduling import group_gaps_by_timesteps, plan_schedule


@pytest.mark.parametrize("num_frames", range(1, 40))
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8])
def test_plan_covers_every_frame_exactly_once(num_frames, k):
    plan = plan_schedule(num_frames, k)
    produced = list(plan.keyframes) + [t for g in plan.gaps for t in g.targets]
    assert sorted(produced) == list(range(num_frames))
    assert len(produced) == len(set(produced)), "a frame is produced twice"


@pytest.mark.parametrize("num_frames", range(1, 40))
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8])
def test_frame_count_is_preserved(num_frames, k):
    plan = plan_schedule(num_frames, k)
    assert len(plan.keyframes) + plan.num_interpolated == num_frames


@pytest.mark.parametrize("num_frames", range(1, 40))
@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_first_and_last_frames_are_always_keyframes(num_frames, k):
    plan = plan_schedule(num_frames, k)
    assert plan.keyframes[0] == 0
    assert plan.keyframes[-1] == num_frames - 1


@pytest.mark.parametrize("num_frames", range(1, 40))
@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_keyframe_gaps_never_exceed_k(num_frames, k):
    """A gap wider than k would mean fewer SR calls than the user asked for."""
    plan = plan_schedule(num_frames, k)
    for a, b in zip(plan.keyframes, plan.keyframes[1:]):
        assert b - a <= k


@pytest.mark.parametrize("num_frames", range(1, 40))
@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_timesteps_are_strictly_inside_the_gap(num_frames, k):
    plan = plan_schedule(num_frames, k)
    for gap in plan.gaps:
        assert len(gap.timesteps) == len(gap.targets)
        assert all(0.0 < t < 1.0 for t in gap.timesteps)
        assert list(gap.timesteps) == sorted(gap.timesteps)
        # timestep t must land on the target's true position in source numbering
        for target, t in zip(gap.targets, gap.timesteps):
            assert gap.start + t * gap.span == pytest.approx(target)


def test_k1_super_resolves_every_frame():
    plan = plan_schedule(10, 1)
    assert plan.keyframes == tuple(range(10))
    assert plan.gaps == ()


def test_regression_tail_shorter_than_k():
    """N=6, K=3 is the case the chunk-based scheduler produced 7 frames for."""
    plan = plan_schedule(6, 3)
    assert plan.keyframes == (0, 3, 5)
    assert [(g.start, g.end, g.targets) for g in plan.gaps] == [
        (0, 3, (1, 2)),
        (3, 5, (4,)),
    ]
    assert plan.gaps[1].timesteps == (0.5,)


def test_regression_tail_of_length_one():
    """N=6, K=2 leaves frame 5 adjacent to keyframe 4 -- no gap, just an SR call."""
    plan = plan_schedule(6, 2)
    assert plan.keyframes == (0, 2, 4, 5)
    assert [g.targets for g in plan.gaps] == [(1,), (3,)]


def test_single_frame_clip():
    plan = plan_schedule(1, 3)
    assert plan.keyframes == (0,)
    assert plan.gaps == ()


def test_two_frame_clip_has_nothing_to_interpolate():
    plan = plan_schedule(2, 3)
    assert plan.keyframes == (0, 1)
    assert plan.gaps == ()


def test_gaps_of_equal_span_batch_together():
    plan = plan_schedule(13, 3)
    grouped = group_gaps_by_timesteps(plan.gaps)
    # every full-length gap shares one bucket
    assert len(grouped) == 1
    assert len(next(iter(grouped.values()))) == 4


def test_tail_gap_batches_separately():
    plan = plan_schedule(12, 3)
    grouped = group_gaps_by_timesteps(plan.gaps)
    assert len(grouped) == 2, "full-length gaps and the short tail need separate batches"


@pytest.mark.parametrize("bad_k", [0, -1])
def test_rejects_invalid_k(bad_k):
    with pytest.raises(ValueError):
        plan_schedule(10, bad_k)


def test_rejects_empty_clip():
    with pytest.raises(ValueError):
        plan_schedule(0, 2)
