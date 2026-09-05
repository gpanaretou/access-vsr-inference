"""Structural and functional checks on the RIFE stage.

The forward-pass tests run anywhere with torch (random weights are enough to
exercise shapes and the pad/warp plumbing). The checkpoint tests are skipped
unless the RIFE weights have already been fetched, so the suite never downloads
anything on its own -- prime them with:

    python -c "from vsr.weights import ensure_interpolation_weights as e; e()"
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vsr.interpolation import InterpolationPipeline
from vsr.interpolation.rife.ifnet import IFNet
from vsr.interpolation.rife.model import RifeModel
from vsr.padding import pad_to_multiple, unpad_tensor
from vsr.weights import DEFAULT_CACHE, INTERPOLATION_RELPATH

_TRAIN_LOG = os.path.join(DEFAULT_CACHE, "interpolation", INTERPOLATION_RELPATH)
_HAS_WEIGHTS = os.path.isfile(os.path.join(_TRAIN_LOG, "flownet.pkl"))
requires_weights = pytest.mark.skipif(
    not _HAS_WEIGHTS, reason="RIFE weights not downloaded"
)


@pytest.mark.parametrize("size", [(64, 64), (128, 192)])
def test_forward_pass_preserves_shape(size):
    model = RifeModel(torch.device("cpu"))
    height, width = size
    img0, img1 = torch.rand(2, 3, height, width), torch.rand(2, 3, height, width)
    out = model.inference(img0, img1, timestep=0.5)
    assert out.shape == img0.shape
    assert torch.isfinite(out).all()


def test_unaligned_resolutions_roundtrip_through_padding():
    model = RifeModel(torch.device("cpu"))
    img0, img1 = torch.rand(1, 3, 100, 140), torch.rand(1, 3, 100, 140)
    padded0, padding = pad_to_multiple(img0, 64)
    padded1, _ = pad_to_multiple(img1, 64)
    out = unpad_tensor(model.inference(padded0, padded1, timestep=0.5), padding)
    assert out.shape == img0.shape


@requires_weights
def test_checkpoint_populates_every_parameter():
    """A rename in the stripped IFNet would show up here as a missing key."""
    state_dict = torch.load(
        os.path.join(_TRAIN_LOG, "flownet.pkl"), map_location="cpu", weights_only=True
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    result = IFNet().load_state_dict(state_dict, strict=False)

    assert result.missing_keys == [], "the checkpoint does not cover every parameter"
    # The only keys we deliberately drop are the training-only distillation
    # teacher and the timestep predictor, neither of which inference reaches.
    assert {k.split(".")[0] for k in result.unexpected_keys} <= {"teacher", "caltime"}


@requires_weights
def test_interpolated_motion_tracks_the_timestep():
    """A square moving left to right should land monotonically between the pair."""
    pipeline = InterpolationPipeline(model_dir=_TRAIN_LOG, device="cpu")

    def square(x):
        frame = np.zeros((128, 128, 3), np.float32)
        frame[48:80, x : x + 32] = 1.0
        return torch.from_numpy(frame).permute(2, 0, 1)[None]

    def centroid_x(tensor):
        column_mass = tensor[0, 0].numpy().sum(0)
        return float(
            (column_mass * np.arange(len(column_mass))).sum() / column_mass.sum()
        )

    img0, img1 = square(10), square(50)
    positions = [
        centroid_x(pipeline(img0, img1, timestep=t)) for t in (0.25, 0.5, 0.75)
    ]

    assert positions == sorted(positions), "interpolated motion is not monotonic in t"
    assert centroid_x(img0) < positions[0]
    assert positions[-1] < centroid_x(img1)
    # RIFE underestimates flow on hard-edged synthetic content; allow real slack.
    for position, expected in zip(positions, (36.0, 46.0, 56.0)):
        assert abs(position - expected) < 5.0
