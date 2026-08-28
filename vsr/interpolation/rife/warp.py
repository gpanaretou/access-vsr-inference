"""Backward warping by optical flow.

Adapted from Practical-RIFE (model/warplayer.py). The grid cache is keyed on
device and dtype as well as size so the module works on any device, rather
than binding to a global CUDA handle at import time.
"""

import torch

_backwarp_grid_cache = {}


def warp(tenInput, tenFlow):
    key = (str(tenFlow.device), str(tenFlow.dtype), str(tenFlow.size()))
    if key not in _backwarp_grid_cache:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device, dtype=tenFlow.dtype)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device, dtype=tenFlow.dtype)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        _backwarp_grid_cache[key] = torch.cat([tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    grid = (_backwarp_grid_cache[key] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=grid, mode="bilinear", padding_mode="border", align_corners=True
    )
