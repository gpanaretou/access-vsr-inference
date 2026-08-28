"""Inference-only RIFE wrapper.

Adapted from Practical-RIFE (train_log/RIFE_HDv3.py) with the optimizer, loss
terms and training/distributed paths removed -- those pulled in torchvision's
VGG weights and an MS-SSIM implementation that inference never touches.
"""

import os

import torch

from .ifnet import IFNet


class RifeModel:
    def __init__(self, device: torch.device):
        self.device = device
        self.flownet = IFNet().to(device)
        self.version = 4.25

    def load(self, model_dir: str) -> "RifeModel":
        path = os.path.join(model_dir, "flownet.pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"RIFE weights not found at {path}. "
                "Call vsr.weights.ensure_interpolation_weights() first."
            )
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.flownet.load_state_dict(state_dict, strict=False)
        self.flownet.eval()
        return self

    @torch.no_grad()
    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        """Synthesizes the frame at `timestep` between img0 and img1.

        img0/img1: (B, 3, H, W) float tensors in [0, 1] with H, W multiples of 64.
        Returns a (B, 3, H, W) tensor.
        """
        imgs = torch.cat((img0, img1), 1)
        scale_list = (16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale)
        _flow, _mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[-1]
