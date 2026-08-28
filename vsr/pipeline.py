"""End-to-end video super-resolution: K-strided diffusion SR + RIFE infill."""

import numpy as np
import torch

from . import frames as frame_io
from . import weights
from .scheduling import group_gaps_by_timesteps, plan_schedule

# The two model stages are imported inside __init__ rather than here: they pull
# in diffusers (and optionally natten), which is a heavy import to pay for
# scheduling-only use, and lets the orchestration be tested against stubs.


def _default_dtype(device: torch.device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class VSRPipeline:
    """Upscales a clip 4x, super-resolving every K-th frame and interpolating the rest.

    The frame count is preserved: N frames in, N frames out. K trades quality
    for speed -- K=1 runs the diffusion model on every frame, K=3 runs it on
    roughly a third of them and synthesizes the other two thirds with RIFE.

    Example:
        pipeline = VSRPipeline(k=2)
        hr_frames = pipeline(lr_frames)   # list of uint8 (H, W, 3) arrays
    """

    def __init__(
        self,
        k: int = 1,
        natten_layers: int = 0,
        natten_kernel_size: int = 7,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        sr_batch_size: int = 4,
        interpolation_batch_size: int = 8,
        compile_decoder: bool = False,
        cache_dir: str = weights.DEFAULT_CACHE,
        sr_model_ckpt: str | None = None,
        sr_decoder_ckpt: str | None = None,
        interpolation_model_dir: str | None = None,
        verbose: bool = True,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = k
        self.sr_batch_size = sr_batch_size
        self.interpolation_batch_size = interpolation_batch_size

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.dtype = dtype or _default_dtype(self.device)

        from .interpolation import InterpolationPipeline
        from .super_resolution import SuperResolutionPipeline

        if sr_model_ckpt is None or sr_decoder_ckpt is None:
            downloaded_model, downloaded_decoder = weights.ensure_super_resolution_weights(cache_dir)
            sr_model_ckpt = sr_model_ckpt or downloaded_model
            sr_decoder_ckpt = sr_decoder_ckpt or downloaded_decoder

        self.sr = SuperResolutionPipeline(
            model_ckpt_path=sr_model_ckpt,
            decoder_ckpt_path=sr_decoder_ckpt,
            device=self.device,
            dtype=self.dtype,
            natten_layers=natten_layers,
            natten_kernel_size=natten_kernel_size,
            compile_decoder=compile_decoder,
            verbose=verbose,
        )
        self.scale_factor = self.sr.scale_factor

        # K=1 never interpolates, so the RIFE weights are not even fetched.
        self.interpolation = None
        if k > 1:
            if interpolation_model_dir is None:
                interpolation_model_dir = weights.ensure_interpolation_weights(cache_dir)
            self.interpolation = InterpolationPipeline(
                model_dir=interpolation_model_dir, device=self.device
            )

    @torch.no_grad()
    def __call__(self, images, k: int | None = None) -> list[np.ndarray]:
        """Super-resolves a clip.

        Args:
            images: list of frames in temporal order. uint8 (H, W, 3) RGB arrays
                are the canonical input; float arrays in [0, 1] and torch
                tensors are also accepted. All frames must share a resolution.
            k: overrides the pipeline's K for this call. Cannot exceed 1 unless
                the pipeline was built with k > 1, since that is what decides
                whether the interpolation model was loaded.

        Returns:
            list of uint8 (4H, 4W, 3) RGB arrays, the same length as `images`.
        """
        k = self.k if k is None else k
        if k > 1 and self.interpolation is None:
            raise ValueError(
                f"k={k} requires the interpolation model, but this pipeline was built "
                f"with k=1. Rebuild it with k>1."
            )

        batch = frame_io.to_batch(images, device=self.device, dtype=self.dtype)
        plan = plan_schedule(num_frames=batch.shape[0], k=k)

        hr = self._super_resolve_keyframes(batch, plan)
        self._interpolate_gaps(hr, plan)

        assert len(hr) == plan.num_frames, (
            f"produced {len(hr)} frames for a {plan.num_frames}-frame clip"
        )
        return [frame_io.to_numpy(hr[i])[0] for i in range(plan.num_frames)]

    def _super_resolve_keyframes(self, batch: torch.Tensor, plan) -> dict[int, torch.Tensor]:
        """Runs the diffusion model over the keyframes in batches of sr_batch_size."""
        hr: dict[int, torch.Tensor] = {}
        keyframes = plan.keyframes

        for start in range(0, len(keyframes), self.sr_batch_size):
            indices = keyframes[start : start + self.sr_batch_size]
            output = self.sr(batch[list(indices)])
            for offset, index in enumerate(indices):
                hr[index] = output[offset : offset + 1]

        return hr

    def _interpolate_gaps(self, hr: dict[int, torch.Tensor], plan) -> None:
        """Fills every non-keyframe in `hr` by interpolating its bounding keyframes.

        Gaps sharing a timestep tuple are batched together, so a clip normally
        needs (K - 1) RIFE calls for the full-length gaps plus a few for the
        shorter tail gap, rather than one call per synthesized frame.
        """
        for timesteps, gaps in group_gaps_by_timesteps(plan.gaps).items():
            for start in range(0, len(gaps), self.interpolation_batch_size):
                chunk = gaps[start : start + self.interpolation_batch_size]
                img0 = torch.cat([hr[gap.start] for gap in chunk])
                img1 = torch.cat([hr[gap.end] for gap in chunk])

                for position, timestep in enumerate(timesteps):
                    output = self.interpolation(img0, img1, timestep=timestep)
                    for offset, gap in enumerate(chunk):
                        hr[gap.targets[position]] = output[offset : offset + 1]
