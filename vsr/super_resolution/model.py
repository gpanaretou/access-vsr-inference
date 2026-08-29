"""Architecture surgery for the AdcSR-style pruned SD-2.1 UNet + half-decoder.

The surgery below (pruning to 75% channels, stripping time embeddings and
cross-attention, swapping in the custom block forwards) must match the surgery
used at training time exactly, or the checkpoint will not load onto it. Treat it
as load-bearing and avoid "cleaning it up".

`vsr/super_resolution/pipeline.py` drives this model at inference time.
"""

import copy
import types

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.downsampling import Downsample2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)
from diffusers.models.upsampling import Upsample2D
from torch import nn

from .forward import (
    MyCrossAttnDownBlock2D_SD_forward,
    MyCrossAttnUpBlock2D_SD_forward,
    MyDownBlock2D_SD_forward,
    MyResnetBlock2D_SD_forward,
    MyTransformer2DModel_SD_forward,
    MyUNet2DConditionModel_SD_forward,
    MyUNetMidBlock2DCrossAttn_SD_forward,
    MyUpBlock2D_SD_forward,
)

BASE_UNET_REPO = "sd-research/stable-diffusion-2-1-base"
SCALE_FACTOR = 4
PAD_DIVISOR = 16
PRUNE_FACTOR = 0.75


def _find_parent(model, module_name):
    components = module_name.split(".")
    parent = model
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    return parent, components[-1]


def _halve_channels(model):
    """Prunes conv/linear/norm layers to PRUNE_FACTOR of their channel count."""
    for name, module in model.named_modules():
        if hasattr(module, "pruned"):
            continue

        if isinstance(module, nn.Conv2d):
            in_channels = int(module.in_channels * PRUNE_FACTOR)
            out_channels = int(module.out_channels * PRUNE_FACTOR)
            new_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation,
                groups=module.groups, bias=module.bias is not None,
            )
            with torch.inference_mode():
                new_conv.weight.copy_(module.weight[:out_channels, :in_channels])
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias[:out_channels])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_conv)
            new_conv.pruned = True
        elif isinstance(module, nn.Linear):
            in_features = int(module.in_features * PRUNE_FACTOR)
            out_features = int(module.out_features * PRUNE_FACTOR)
            new_linear = nn.Linear(
                in_features=in_features, out_features=out_features,
                bias=module.bias is not None,
            )
            with torch.inference_mode():
                new_linear.weight.copy_(module.weight[:out_features, :in_features])
                if module.bias is not None:
                    new_linear.bias.copy_(module.bias[:out_features])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_linear)
            new_linear.pruned = True
        elif isinstance(module, nn.GroupNorm):
            num_channels = int(module.num_channels * PRUNE_FACTOR)
            for num_groups in [32, 24, 16, 12, 8, 6, 4, 2, 1]:
                if num_channels % num_groups == 0:
                    break
            new_gn = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels,
                eps=module.eps, affine=module.affine,
            )
            with torch.inference_mode():
                new_gn.weight.copy_(module.weight[:num_channels])
                new_gn.bias.copy_(module.bias[:num_channels])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_gn)
            new_gn.pruned = True
        elif isinstance(module, nn.LayerNorm):
            normalized_shape = int(module.normalized_shape[0] * PRUNE_FACTOR)
            new_ln = nn.LayerNorm(
                normalized_shape, eps=module.eps,
                elementwise_affine=module.elementwise_affine,
            )
            with torch.inference_mode():
                new_ln.weight.copy_(module.weight[:normalized_shape])
                new_ln.bias.copy_(module.bias[:normalized_shape])
            parent, last_name = _find_parent(model, name)
            setattr(parent, last_name, new_ln)
            new_ln.pruned = True
        elif isinstance(module, (Downsample2D, Upsample2D)):
            module.channels = int(module.channels * PRUNE_FACTOR)
    return model


def _prepare_pruned_unet(unet):
    """Strips the UNet to an unconditional single-step feature extractor."""
    if hasattr(unet, "time_embedding"):
        del unet.time_embedding

    unet.apply(lambda m: m.register_forward_pre_hook(lambda module, p: p[0].to(unet.device)))

    new_conv_in = nn.Conv2d(16, 320, 3, padding=1)
    new_conv_in.weight.data = unet.conv_in.weight.data.repeat(1, 4, 1, 1)
    new_conv_in.bias.data = unet.conv_in.bias.data
    unet.conv_in = new_conv_in

    new_conv_out = nn.Conv2d(320, 342, 3, padding=1)
    new_conv_out.weight.data = unet.conv_out.weight.data.repeat(86, 1, 1, 1)[:342]
    new_conv_out.bias.data = unet.conv_out.bias.data.repeat(86)[:342]
    unet.conv_out = new_conv_out

    def remove_unused_layers(module):
        if isinstance(module, ResnetBlock2D) and hasattr(module, "time_emb_proj"):
            del module.time_emb_proj
        if isinstance(module, BasicTransformerBlock):
            if hasattr(module, "attn2"):
                del module.attn2
            if hasattr(module, "norm2"):
                del module.norm2

    unet.apply(remove_unused_layers)

    def replace_forward_methods(module):
        if isinstance(module, CrossAttnDownBlock2D):
            module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_forward, module)
        elif isinstance(module, DownBlock2D):
            module.forward = types.MethodType(MyDownBlock2D_SD_forward, module)
        elif isinstance(module, UNetMidBlock2DCrossAttn):
            module.forward = types.MethodType(MyUNetMidBlock2DCrossAttn_SD_forward, module)
        elif isinstance(module, UpBlock2D):
            module.forward = types.MethodType(MyUpBlock2D_SD_forward, module)
        elif isinstance(module, CrossAttnUpBlock2D):
            module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_forward, module)
        elif isinstance(module, ResnetBlock2D):
            module.forward = types.MethodType(MyResnetBlock2D_SD_forward, module)
        elif isinstance(module, Transformer2DModel):
            module.forward = types.MethodType(MyTransformer2DModel_SD_forward, module)

    unet.apply(replace_forward_methods)
    unet.forward = types.MethodType(MyUNet2DConditionModel_SD_forward, unet)

    _halve_channels(unet)

    unet.body = nn.Sequential(
        *unet.down_blocks,
        unet.mid_block,
        *unet.up_blocks,
        unet.conv_norm_out,
        unet.conv_act,
        unet.conv_out,
    )
    return unet


class RefactoredNet(nn.Module):
    def __init__(self, unet, mid_block, up_blocks, conv_norm_out, conv_act, conv_out):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.unet = unet
        self.mid_block = mid_block
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_norm_out = conv_norm_out
        self.conv_act = conv_act
        self.conv_out = conv_out

    def decode(self, x):
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        return self.conv_out(x)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.unet(x)
        x = self.mid_block(x)
        return self.decode(x)


def load_decoder(ckpt_path, device, dtype):
    """Loads the VAE decoder whose blocks are grafted onto the pruned UNet."""
    decoder = Decoder(
        in_channels=4, out_channels=3, up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[64, 128, 256, 256], layers_per_block=2,
        norm_num_groups=32, act_fn="silu", norm_type="group",
        mid_block_add_attention=True,
    ).to(device=device, dtype=dtype)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    decoder_state_dict = {
        k.replace("decoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if "decoder" in k
    }
    decoder.load_state_dict(decoder_state_dict, strict=True)
    return decoder


def load_model_weights(model, ckpt_path):
    """Remaps the training-time `body.N.` keys onto RefactoredNet's submodules."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    remapped = {}
    for key, value in checkpoint.items():
        if key.startswith("body.0."):
            new_key = key.replace("body.0.", "pixel_unshuffle.")
        elif key.startswith("body.1."):
            new_key = key.replace("body.1.", "unet.", 1)
        elif key.startswith("body.2."):
            new_key = key.replace("body.2.", "mid_block.")
        else:
            new_key = key
        remapped[new_key] = value

    model.load_state_dict(remapped, strict=False)


def build_model(model_ckpt_path, decoder_ckpt_path, device, dtype, log=lambda *a: None):
    """Builds a RefactoredNet with the SR checkpoint loaded, in eval mode."""
    log("[vsr] building pruned UNet...")
    base_unet = StableDiffusionPipeline.from_pretrained(
        BASE_UNET_REPO, torch_dtype=dtype
    ).unet.to(device)
    pruned_unet = _prepare_pruned_unet(base_unet)
    del base_unet

    log("[vsr] loading decoder...")
    decoder = load_decoder(decoder_ckpt_path, device, dtype)
    mid_block = copy.deepcopy(decoder.mid_block)
    up_blocks = copy.deepcopy(decoder.up_blocks)
    conv_norm_out = copy.deepcopy(decoder.conv_norm_out)
    conv_act = copy.deepcopy(decoder.conv_act)
    conv_out = copy.deepcopy(decoder.conv_out)
    del decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = RefactoredNet(
        pruned_unet, mid_block, up_blocks, conv_norm_out, conv_act, conv_out
    ).to(device=device, dtype=dtype)

    log(f"[vsr] loading SR weights from {model_ckpt_path}")
    load_model_weights(model, model_ckpt_path)
    model.eval()
    return model
