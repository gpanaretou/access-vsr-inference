"""End-to-end routing checks with stub models -- no weights, no GPU.

Frame i is filled with the constant value i/255. The SR stub preserves that
value, and the interpolation stub returns a linear blend of its two inputs.
Because a*(1-t) + b*t == a + t*(b-a) == the target's true source index, output
frame j is a solid j/255 if and only if every keyframe, gap assignment and
timestep is routed correctly. Any misrouting shows up as a wrong pixel value.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vsr.pipeline import VSRPipeline


class _SRStub:
    scale_factor = 4

    def __init__(self):
        self.calls = []

    def __call__(self, batch):
        self.calls.append(batch.shape[0])
        return batch.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)


class _InterpolationStub:
    def __init__(self):
        self.calls = []

    def __call__(self, img0, img1, timestep, scale=1.0):
        self.calls.append((img0.shape[0], timestep))
        return img0 * (1.0 - timestep) + img1 * timestep


def _build(k, batch_size=4):
    pipeline = object.__new__(VSRPipeline)
    pipeline.k = k
    pipeline.batch_size = batch_size
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.sr = _SRStub()
    pipeline.interpolation = _InterpolationStub()
    pipeline.scale_factor = 4
    return pipeline


def _ramp_clip(num_frames, height=8, width=8):
    return [np.full((height, width, 3), i, dtype=np.uint8) for i in range(num_frames)]


@pytest.mark.parametrize("num_frames", [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 17])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_output_length_matches_input_length(num_frames, k):
    pipeline = _build(k)
    out = pipeline(_ramp_clip(num_frames))
    assert len(out) == num_frames


@pytest.mark.parametrize("num_frames", [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 17])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_every_frame_lands_at_its_own_temporal_position(num_frames, k):
    pipeline = _build(k)
    out = pipeline(_ramp_clip(num_frames))
    for index, frame in enumerate(out):
        assert frame.shape == (32, 32, 3)
        np.testing.assert_array_equal(
            frame,
            np.full((32, 32, 3), index, dtype=np.uint8),
            err_msg=f"frame {index} of a {num_frames}-frame clip (k={k}) is misrouted",
        )


def test_k_reduces_the_number_of_diffusion_calls():
    frames = _ramp_clip(13)

    k1 = _build(1)
    k1(frames)
    k3 = _build(3)
    k3(frames)

    assert sum(k1.sr.calls) == 13
    assert sum(k3.sr.calls) == 5  # keyframes 0, 3, 6, 9, 12


def test_interpolation_is_batched_across_gaps():
    """One RIFE call per (timestep, gap-bucket), not one per synthesized frame."""
    pipeline = _build(3)
    pipeline(_ramp_clip(13))
    # 4 full-length gaps in one bucket, 2 timesteps each
    assert pipeline.interpolation.calls == [(4, 1 / 3), (4, 2 / 3)]


def test_interpolation_batch_size_is_respected():
    pipeline = _build(3, batch_size=2)
    pipeline(_ramp_clip(13))
    assert all(batch <= 2 for batch, _ in pipeline.interpolation.calls)


def test_sr_batch_size_is_respected():
    pipeline = _build(1, batch_size=3)
    pipeline(_ramp_clip(10))
    assert pipeline.sr.calls == [3, 3, 3, 1]


def test_k1_never_touches_the_interpolation_model():
    pipeline = _build(1)
    pipeline(_ramp_clip(9))
    assert pipeline.interpolation.calls == []


def test_per_call_k_override():
    pipeline = _build(3)
    pipeline(_ramp_clip(13), k=1)
    assert sum(pipeline.sr.calls) == 13
    assert pipeline.interpolation.calls == []


def test_k_override_above_one_requires_a_loaded_interpolation_model():
    pipeline = _build(1)
    pipeline.interpolation = None
    with pytest.raises(ValueError, match="requires the interpolation model"):
        pipeline(_ramp_clip(9), k=3)
