"""Streaming must be indistinguishable from processing the whole clip at once."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_pipeline_routing import _build, _ramp_clip

from vsr.scheduling import is_keyframe, plan_schedule
from vsr.streaming import VSRStream


def _stream_clip(pipeline, clip, batch_size):
    stream = VSRStream(pipeline)
    out = []
    for start in range(0, len(clip), batch_size):
        out.extend(stream.push(clip[start : start + batch_size]))
    out.extend(stream.flush())
    return out


@pytest.mark.parametrize("num_frames", [1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 20, 24])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 8])
def test_streaming_matches_whole_clip_exactly(num_frames, k, batch_size):
    clip = _ramp_clip(num_frames)

    whole = _build(k)(clip)
    streamed = _stream_clip(_build(k), clip, batch_size)

    assert len(streamed) == len(whole) == num_frames
    for index, (a, b) in enumerate(zip(streamed, whole)):
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index} differs")


@pytest.mark.parametrize("num_frames", [1, 2, 3, 6, 13, 24])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_every_frame_lands_at_its_own_position(num_frames, k, batch_size):
    streamed = _stream_clip(_build(k), _ramp_clip(num_frames), batch_size)
    for index, frame in enumerate(streamed):
        np.testing.assert_array_equal(
            frame, np.full((32, 32, 3), index, dtype=np.uint8),
            err_msg=f"frame {index} misrouted (N={num_frames}, k={k}, batch={batch_size})",
        )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 6, 8])
def test_keyframe_stride_survives_batch_boundaries(batch_size):
    """The whole point: batching must not force extra keyframes."""
    pipeline = _build(2)
    stream = VSRStream(pipeline)
    clip = _ramp_clip(24)
    for start in range(0, 24, batch_size):
        stream.push(clip[start : start + batch_size])
    stream.flush()

    # 24 frames at k=2 -> keyframes 0,2,...,22 plus the final frame 23
    assert sum(pipeline.sr.calls) == len(plan_schedule(24, 2).keyframes) == 13


def test_per_batch_calls_would_defeat_k():
    """Contrast: calling the pipeline per batch super-resolves every frame."""
    naive = _build(2)
    clip = _ramp_clip(24)
    for start in range(0, 24, 2):
        naive(clip[start : start + 2])
    assert sum(naive.sr.calls) == 24  # k=2 bought nothing

    streamed = _build(2)
    _stream_clip(streamed, clip, 2)
    assert sum(streamed.sr.calls) == 13


def test_frames_are_buffered_until_the_next_keyframe_arrives():
    """A non-keyframe cannot be emitted before its right anchor exists."""
    clip = _ramp_clip(4)
    stream = VSRStream(_build(3))

    # Frame 0 is a keyframe and the leftmost anchor -- it needs no right anchor.
    assert len(stream.push(clip[0:1])) == 1

    # Frames 1 and 2 sit between keyframes 0 and 3, which has not arrived yet.
    assert stream.push(clip[1:3]) == []
    assert stream.frames_buffered == 2

    # Keyframe 3 arrives and releases all three.
    assert len(stream.push(clip[3:4])) == 3
    assert stream.frames_buffered == 0
    assert stream.frames_received == 4


def test_counters_track_progress():
    stream = VSRStream(_build(2))
    clip = _ramp_clip(9)
    total = 0
    for start in range(0, 9, 2):
        total += len(stream.push(clip[start : start + 2]))
    assert stream.frames_received == 9
    total += len(stream.flush())
    assert total == 9


def test_reset_clears_carryover_between_clips():
    pipeline = _build(2)
    stream = VSRStream(pipeline)
    stream.push(_ramp_clip(5))
    stream.flush()  # flush resets

    assert stream.frames_received == 0
    assert stream.frames_buffered == 0
    second = []
    for start in range(0, 5, 2):
        second.extend(stream.push(_ramp_clip(5)[start : start + 2]))
    second.extend(stream.flush())
    assert len(second) == 5


def test_shape_change_mid_stream_is_rejected():
    stream = VSRStream(_build(2))
    stream.push([np.zeros((8, 8, 3), np.uint8)])
    with pytest.raises(ValueError, match="frame shape changed"):
        stream.push([np.zeros((16, 16, 3), np.uint8)])


def test_empty_push_is_a_noop():
    stream = VSRStream(_build(2))
    assert stream.push([]) == []
    assert stream.frames_received == 0


def test_stream_requires_interpolation_model_for_k_above_one():
    pipeline = _build(1)
    pipeline.interpolation = None
    with pytest.raises(ValueError, match="requires the interpolation model"):
        VSRStream(pipeline, k=2)


def test_keyframe_membership_is_index_based_not_batch_based():
    assert [i for i in range(10) if is_keyframe(i, 3)] == [0, 3, 6, 9]
