import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vsr import frames as frame_io
from vsr.padding import pad_to_multiple, scale_padding, unpad_tensor


def test_uint8_hwc_roundtrips_exactly():
    rng = np.random.default_rng(0)
    original = [rng.integers(0, 256, (12, 20, 3), dtype=np.uint8) for _ in range(3)]

    batch = frame_io.to_batch(original, device="cpu", dtype=torch.float32)
    assert batch.shape == (3, 3, 12, 20)

    restored = frame_io.to_numpy(batch)
    for before, after in zip(original, restored):
        assert after.dtype == np.uint8
        assert after.shape == (12, 20, 3)
        np.testing.assert_array_equal(before, after)


def test_chw_input_is_accepted():
    chw = np.zeros((3, 12, 20), dtype=np.uint8)
    batch = frame_io.to_batch([chw], device="cpu", dtype=torch.float32)
    assert batch.shape == (1, 3, 12, 20)


def test_float_frames_in_unit_range_are_accepted():
    batch = frame_io.to_batch(
        [np.full((8, 8, 3), 0.5, dtype=np.float32)], device="cpu", dtype=torch.float32
    )
    assert torch.allclose(batch, torch.full_like(batch, 0.5))


def test_float_frames_outside_unit_range_are_rejected():
    """0-255 floats would be silently 255x too bright rather than erroring."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        frame_io.to_batch(
            [np.full((8, 8, 3), 200.0, dtype=np.float32)],
            device="cpu",
            dtype=torch.float32,
        )


def test_ragged_clips_are_rejected():
    with pytest.raises(ValueError, match="share a shape"):
        frame_io.to_batch(
            [np.zeros((8, 8, 3), np.uint8), np.zeros((9, 8, 3), np.uint8)],
            device="cpu",
            dtype=torch.float32,
        )


def test_empty_clip_is_rejected():
    with pytest.raises(ValueError):
        frame_io.to_batch([], device="cpu", dtype=torch.float32)


def test_out_of_range_output_is_clamped_not_wrapped():
    out = frame_io.to_numpy(
        torch.tensor([[[[-1.0, 2.0]]]]).expand(1, 3, 1, 2).contiguous()
    )
    assert out[0].min() == 0 and out[0].max() == 255


@pytest.mark.parametrize("shape", [(1, 3, 17, 23), (1, 3, 16, 16), (2, 3, 180, 320)])
@pytest.mark.parametrize("divisor", [16, 64])
def test_padding_roundtrips(shape, divisor):
    x = torch.rand(shape)
    padded, padding = pad_to_multiple(x, divisor)
    assert padded.shape[-2] % divisor == 0
    assert padded.shape[-1] % divisor == 0
    assert torch.equal(unpad_tensor(padded, padding), x)


def test_scaled_padding_unpads_a_4x_upsampled_tensor():
    x = torch.rand(1, 3, 45, 79)
    padded, padding = pad_to_multiple(x, 16)
    upsampled = torch.rand(1, 3, padded.shape[-2] * 4, padded.shape[-1] * 4)
    cropped = unpad_tensor(upsampled, scale_padding(padding, 4))
    assert cropped.shape == (1, 3, 45 * 4, 79 * 4)


def test_small_frames_fall_back_from_reflect_padding():
    """Reflect padding cannot pad 16px up to 64; the fallback must still roundtrip."""
    x = torch.rand(1, 3, 16, 16)
    padded, padding = pad_to_multiple(x, 64)
    assert padded.shape == (1, 3, 64, 64)
    assert torch.equal(unpad_tensor(padded, padding), x)


def test_reflect_is_still_used_when_it_fits():
    x = torch.rand(1, 3, 180, 320)
    padded, _ = pad_to_multiple(x, 64)
    expected = torch.nn.functional.pad(x, (0, 0, 6, 6), mode="reflect")
    assert torch.equal(padded, expected)


def test_to_numpy_does_not_mutate_its_input():
    """Tensor.float() is a no-op for fp32, so in-place scaling would corrupt the source."""
    original = torch.rand(1, 3, 4, 4)
    snapshot = original.clone()
    frame_io.to_numpy(original)
    assert torch.equal(original, snapshot)


@pytest.mark.parametrize("channels", [1, 4])
def test_non_rgb_frames_are_rejected(channels):
    with pytest.raises(ValueError, match="3-channel RGB"):
        frame_io.to_batch(
            [np.zeros((8, 8, channels), np.uint8)], device="cpu", dtype=torch.float32
        )
