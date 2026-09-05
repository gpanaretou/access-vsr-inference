"""RIFE frame interpolation stage."""

import torch

from ..padding import pad_to_multiple, unpad_tensor
from .rife import RifeModel

# IFNet downsamples by 2 five times over the pyramid; 64 keeps every level aligned.
PAD_DIVISOR = 64


class InterpolationPipeline:
    """Synthesizes in-between frames from pairs of HR keyframes.

    Runs in float32 regardless of the SR stage's dtype: RIFE is cheap relative
    to the diffusion model, and its flow estimation is the part of the pipeline
    most sensitive to reduced precision.
    """

    def __init__(self, model_dir: str, device="cuda"):
        self.device = torch.device(device)
        self.model = RifeModel(self.device).load(model_dir)

    @torch.no_grad()
    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float,
        scale: float = 1.0,
    ):
        """Interpolates one frame per (img0, img1) pair in the batch.

        img0/img1: (B, 3, H, W) tensors in [0, 1] of any resolution.
        timestep: position in [0, 1] between img0 and img1.
        Returns a (B, 3, H, W) float32 tensor at the input resolution.
        """
        assert img0.shape == img1.shape, (
            f"keyframe pair must share a shape, got {tuple(img0.shape)} and {tuple(img1.shape)}"
        )

        img0 = img0.to(device=self.device, dtype=torch.float32)
        img1 = img1.to(device=self.device, dtype=torch.float32)

        padded0, padding = pad_to_multiple(img0, PAD_DIVISOR)
        padded1, _ = pad_to_multiple(img1, PAD_DIVISOR)

        out = self.model.inference(padded0, padded1, timestep=timestep, scale=scale)
        return unpad_tensor(out, padding).clamp_(0.0, 1.0)
