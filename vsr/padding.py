"""Reflect-padding helpers shared by the SR and interpolation stages."""

import torch
import torch.nn.functional as F


def pad_to_multiple(image_tensor, divisor=16):
    """Reflect-pads a (B,C,H,W) tensor so H and W are divisible by `divisor`.

    Returns (padded_tensor, padding) where padding is the F.pad 4-tuple
    (left, right, top, bottom), suitable for unpad_tensor.
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    assert image_tensor.ndim == 4, (
        f"expected tensor of shape [Batch, Channels, Height, Width], got {image_tensor.shape}"
    )

    height = image_tensor.shape[-2]
    width = image_tensor.shape[-1]

    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)

    # Reflect padding requires every pad to be smaller than its dimension, which
    # a small frame can violate (e.g. a 16px side padded to a multiple of 64).
    # Replicate has no such limit; it is only reached where reflect would have
    # raised, so results are unchanged for any size that already worked.
    mode = "reflect" if max(pad_h, pad_w) < min(height, width) else "replicate"
    return F.pad(image_tensor, padding, mode=mode), padding


def unpad_tensor(padded_tensor, padding):
    """Inverse of pad_to_multiple for a (B,C,H,W) tensor."""
    assert padded_tensor.ndim == 4, (
        f"expected tensor of shape [Batch, Channels, Height, Width], got {padded_tensor.shape}"
    )

    pad_left, pad_right, pad_top, pad_bottom = padding
    height_padded = padded_tensor.shape[-2]
    width_padded = padded_tensor.shape[-1]

    return padded_tensor[
        :,
        :,
        pad_top : height_padded - pad_bottom,
        pad_left : width_padded - pad_right,
    ]


def scale_padding(padding, factor):
    """Scales a padding 4-tuple by an integer upsampling factor."""
    return tuple(p * factor for p in padding)
