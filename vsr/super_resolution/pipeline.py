"""Single-step diffusion super-resolution stage (AdcSR-style pruned SD-2.1 UNet).

The architecture surgery below (pruning to 75% channels, stripping time
embeddings and cross-attention, swapping in the custom block forwards) must
match the surgery used at training time exactly, or the checkpoint will not
load onto it. Treat it as load-bearing and avoid "cleaning it up".
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

from ..padding import pad_to_multiple, scale_padding, unpad_tensor
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


class SuperResolutionPipeline:
    """4x single-step diffusion super-resolution over a batch of LR frames."""

    def __init__(
        self,
        model_ckpt_path: str,
        decoder_ckpt_path: str,
        device="cuda",
        dtype=torch.bfloat16,
        natten_layers: int = 0,
        natten_kernel_size: int = 7,
        compile_decoder: bool = False,
        verbose: bool = True,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.scale_factor = SCALE_FACTOR
        self._natten_applied = False
        self._log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

        self._log("[vsr] building pruned UNet...")
        base_unet = StableDiffusionPipeline.from_pretrained(
            BASE_UNET_REPO, torch_dtype=self.dtype
        ).unet.to(self.device)
        pruned_unet = _prepare_pruned_unet(base_unet)
        del base_unet

        self._log("[vsr] loading decoder...")
        decoder = self._load_decoder(decoder_ckpt_path)
        mid_block = copy.deepcopy(decoder.mid_block)
        up_blocks = copy.deepcopy(decoder.up_blocks)
        conv_norm_out = copy.deepcopy(decoder.conv_norm_out)
        conv_act = copy.deepcopy(decoder.conv_act)
        conv_out = copy.deepcopy(decoder.conv_out)
        del decoder
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.model = RefactoredNet(
            pruned_unet, mid_block, up_blocks, conv_norm_out, conv_act, conv_out
        ).to(device=self.device, dtype=self.dtype)

        self._log(f"[vsr] loading SR weights from {model_ckpt_path}")
        self._load_model_weights(model_ckpt_path)
        self.model.eval()

        if natten_layers > 0:
            self.use_natten_attention(
                natten_layers=natten_layers, kernel_size=natten_kernel_size
            )

        if compile_decoder:
            # Recompiles on every new input resolution -- only worth it for a
            # fixed-resolution workload with many frames.
            self.model.decode = torch.compile(self.model.decode)

        self._log("[vsr] super-resolution pipeline ready")

    def _load_decoder(self, ckpt_path):
        decoder = Decoder(
            in_channels=4, out_channels=3, up_block_types=["UpDecoderBlock2D"] * 4,
            block_out_channels=[64, 128, 256, 256], layers_per_block=2,
            norm_num_groups=32, act_fn="silu", norm_type="group",
            mid_block_add_attention=True,
        ).to(device=self.device, dtype=self.dtype)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        decoder_state_dict = {
            k.replace("decoder.", ""): v
            for k, v in ckpt["state_dict"].items()
            if "decoder" in k
        }
        decoder.load_state_dict(decoder_state_dict, strict=True)
        return decoder

    def _load_model_weights(self, ckpt_path):
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

        self.model.load_state_dict(remapped, strict=False)

    def use_natten_attention(
        self,
        natten_layers: int = 3,
        kernel_size: int = 7,
        dilation: int = 1,
        stride: int = 1,
        stage: str = "up",
        down_layers: int = 0,
    ):
        """Swaps the highest-resolution (240-dim) attn1 processors for NATTEN.

        body.8 is the UNet's upsampling decoder (3 attention layers at 240 dim);
        body.0 is the downsampling encoder (2 layers at 240 dim). This is a
        one-way toggle -- build a fresh pipeline to change the layer count.

        If the `natten` package is not installed the swap is skipped and the
        pipeline keeps its standard attention processors.
        """
        if self._natten_applied:
            raise RuntimeError(
                "NATTEN attention has already been applied; build a new pipeline to change it"
            )
        assert stage in ("up", "down"), f"stage must be 'up' or 'down', got {stage}"

        stage_counts = {"body.8": [3, 0], "body.0": [2, 0]}
        stage_counts["body.8" if stage == "up" else "body.0"][1] = natten_layers
        if down_layers > 0:
            assert stage == "up", (
                "down_layers combines a downsampling replacement with an upsampling one; "
                "when stage='down' pass the downsampling count via natten_layers instead"
            )
            stage_counts["body.0"][1] = down_layers

        for target_body, (max_layers, count) in stage_counts.items():
            assert count <= max_layers, (
                f"cannot exceed {max_layers} layers for {target_body}, got {count}"
            )

        try:
            from .natten_processor import AutoNattenAttentionProcessor2D
        except ImportError as exc:
            # Keeping the stock AttnProcessor2_0 processors is a valid config,
            # just slower at high resolutions -- no reason to hard-fail here.
            print(
                f"[vsr] natten is unavailable ({exc}); falling back to standard "
                "attention. Install it with: "
                "uv pip install natten==0.21.5+torch2100cu128 -f https://whl.natten.org"
            )
            return

        natten_processor = AutoNattenAttentionProcessor2D(
            kernel_size=kernel_size, dilation=dilation, stride=stride
        )

        attn_procs = self.model.unet.attn_processors
        new_attn_procs = {}
        counters = {"body.8": 0, "body.0": 0}
        replaced = 0

        for name, proc in attn_procs.items():
            assert "attn1" in name
            attn_module = self.model.unet.get_submodule(name.replace(".processor", ""))
            target_body = next((b for b in stage_counts if b in name), None)

            if attn_module.to_q.in_features == 240 and target_body is not None:
                counters[target_body] += 1
                if counters[target_body] <= stage_counts[target_body][1]:
                    new_attn_procs[name] = natten_processor
                    replaced += 1
                else:
                    new_attn_procs[name] = proc
            else:
                new_attn_procs[name] = proc

        self._log(
            f"[vsr] replaced {replaced}/{len(attn_procs)} attention processors with NATTEN "
            f"(up={stage_counts['body.8'][1]}, down={stage_counts['body.0'][1]})"
        )
        self.model.unet.set_attn_processor(new_attn_procs)
        self._natten_applied = True

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Super-resolves a (B, 3, H, W) float tensor in [0, 1] by 4x.

        Returns a (B, 3, 4H, 4W) tensor on the pipeline's device and dtype.
        """
        assert batch.ndim == 4, f"expected (B, C, H, W), got {tuple(batch.shape)}"

        batch = batch.to(device=self.device, dtype=self.dtype)
        padded, padding = pad_to_multiple(batch, PAD_DIVISOR)
        output = self.model(padded)
        return unpad_tensor(output, scale_padding(padding, SCALE_FACTOR))
