"""Single-step diffusion super-resolution stage (AdcSR-style pruned SD-2.1 UNet).

The model itself -- the architecture surgery and checkpoint loading -- lives in
`model.py`; this module only drives it at inference time.
"""

import torch

from ..padding import pad_to_multiple, scale_padding, unpad_tensor
from .model import PAD_DIVISOR, SCALE_FACTOR, build_model


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

        self.model = build_model(
            model_ckpt_path, decoder_ckpt_path, self.device, self.dtype, self._log
        )

        if natten_layers > 0:
            self.use_natten_attention(
                natten_layers=natten_layers, kernel_size=natten_kernel_size
            )

        if compile_decoder:
            # Recompiles on every new input resolution -- only worth it for a
            # fixed-resolution workload with many frames.
            self.model.decode = torch.compile(self.model.decode)

        self._log("[vsr] super-resolution pipeline ready")

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
