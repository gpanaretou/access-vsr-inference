"""Conversion between the public numpy frame format and internal tensors.

Public format: uint8 RGB arrays of shape (H, W, 3). Float arrays in [0, 1] and
torch tensors are also accepted on input as a convenience; output is always
uint8 (H, W, 3) so callers can hand frames straight to imageio/OpenCV/PIL.
"""

import numpy as np
import torch

_CHANNEL_COUNTS = (1, 3, 4)


def _array_to_chw(array: np.ndarray) -> np.ndarray:
    """Normalizes an HWC or CHW array to CHW, rejecting ambiguous shapes."""
    if array.ndim != 3:
        raise ValueError(f"expected a 3-dimensional frame, got shape {array.shape}")

    hwc = array.shape[-1] in _CHANNEL_COUNTS
    chw = array.shape[0] in _CHANNEL_COUNTS

    if hwc and chw:
        # e.g. (3, 3, 3) -- genuinely ambiguous, so pick the documented layout.
        return np.transpose(array, (2, 0, 1))
    if hwc:
        return np.transpose(array, (2, 0, 1))
    if chw:
        return array
    raise ValueError(
        f"cannot infer channel axis for frame of shape {array.shape}; "
        f"expected (H, W, C) or (C, H, W) with C in {_CHANNEL_COUNTS}"
    )


def _to_float_chw_tensor(frame) -> torch.Tensor:
    """Converts one frame to a (C, H, W) float tensor in [0, 1]."""
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"batched tensor frames must have batch size 1, got {tensor.shape}"
                )
            tensor = tensor.squeeze(0)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        else:
            tensor = tensor.float()
        return tensor

    array = np.asarray(frame)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
        if array.size and float(array.max()) > 1.5:
            raise ValueError(
                "float frames must be in [0, 1]; got values above 1.0. "
                "Pass a uint8 array instead if these are 0-255 pixels."
            )

    return torch.from_numpy(np.ascontiguousarray(_array_to_chw(array)))


def to_batch(frames, device, dtype) -> torch.Tensor:
    """Stacks a list of frames into one (N, C, H, W) tensor on `device`.

    All frames must share a resolution -- a video pipeline has no meaningful
    behaviour for a ragged clip, and silently resizing would corrupt output.
    """
    if not frames:
        raise ValueError("expected at least one frame")

    tensors = [_to_float_chw_tensor(f) for f in frames]

    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"all frames must share a shape, got {sorted(shapes)}")

    channels = tensors[0].shape[0]
    if channels != 3:
        raise ValueError(
            f"expected 3-channel RGB frames, got {channels} channels. "
            "Convert grayscale to RGB and drop any alpha channel before calling."
        )

    return torch.stack(tensors).to(device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> list[np.ndarray]:
    """Converts a (N, C, H, W) or (1, C, H, W) tensor to uint8 (H, W, C) arrays."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        
    array = (
        tensor.detach().to(torch.float32).clamp(0.0, 1.0).mul(255.0).round()
        .to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    )
    return [np.ascontiguousarray(frame) for frame in array]
