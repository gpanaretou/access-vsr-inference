"""NATTEN neighborhood-attention drop-in for the UNet's attn1 processors."""

import math

import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
from natten import na2d


class AutoNattenAttentionProcessor2D(AttnProcessor2_0):
    """
    Processor using NATTEN NeighborhoodAttention2D.
    """

    def __init__(self, kernel_size=5, dilation=1, stride=2, aspect_ratio=1.0):
        # dilation=0 means "derive it from the first feature map this processor
        # sees", which then sticks for every other layer and resolution sharing
        # the instance. Pass an explicit dilation to avoid that.
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.aspect_ratio = aspect_ratio

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            return attn.processor.__class__()(
                attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )

        bsz, seq_len, dim = hidden_states.shape

        W = max(1, round(math.sqrt(seq_len / self.aspect_ratio)))
        H = math.ceil(seq_len / W)

        target_seq_len = H * W

        if not self.dilation:
            self.dilation = min(H, W) // self.kernel_size

        pad_len = target_seq_len - seq_len
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        scale = attn.scale if attn.scale is not None else (dim // attn.heads) ** -0.5
        query = query * scale

        head_dim = dim // attn.heads
        query = query.view(bsz, H, W, attn.heads, head_dim).contiguous()
        key = key.view(bsz, H, W, attn.heads, head_dim).contiguous()
        value = value.view(bsz, H, W, attn.heads, head_dim).contiguous()

        out = na2d(
            query,
            key,
            value,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            stride=self.stride,
        )

        out = out.view(bsz, target_seq_len, dim)

        if pad_len > 0:
            out = out[:, :seq_len, :]

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out
