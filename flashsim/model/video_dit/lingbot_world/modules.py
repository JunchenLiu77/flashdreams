import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from flashsim.model.video_dit.wan2_1.modules import (
    BlockCache,
    SelfAttention,
    CrossAttention,
)


class Block(nn.Module):
    """Transformer block with self-attn, cross-attn, and FFN branches."""

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Core submodules
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = SelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
        )
        self.norm3 = (
            nn.LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = CrossAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context_text: Tensor,
        context_img: Tensor | None = None,
    ) -> BlockCache:
        """Initialize per-branch caches for this transformer block.

        Args:
            chunk_size: Number of tokens appended per streaming update step.
            window_size: Rolling-window capacity in tokens.
            sink_size: Sink-token capacity retained permanently.
            context_text: Text context tensor [..., L_text, D].
            context_img: Optional image context tensor [..., L_img, D] for I2V.

        Returns:
            ``BlockCache`` initialized for this block.
        """
        batch_shape = context_text.shape[:-2]
        batch_size = math.prod(batch_shape)
        device = context_text.device
        dtype = context_text.dtype

        return BlockCache(
            self_attn=self.self_attn.initialize_cache(
                batch_size,
                chunk_size,
                window_size,
                sink_size,
                device=device,
                dtype=dtype,
            ),
            cross_attn=self.cross_attn.initialize_cache(context_text, context_img),
        )

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Set context-parallel process group for self-attention."""
        self.self_attn.set_context_parallel_group(cp_group)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        plucker_embedding: Tensor | None = None,
    ) -> Tensor:
        """Run one transformer block update.

        Args:
            x: Input tensor with shape [..., L, D].
            e: Modulation tensor with shape [..., 6, D].
            cache: KV cache container for this block.
            rope_freqs: RoPE frequencies with shape [L, 1, 1, head_dim // 2].
            plucker_embedding: Optional Camera Control.Plucker embedding of
                shape [..., (L1+...+Ln), D], camera-to-world space.

        Returns:
            Updated hidden states with shape [..., L, D].
        """
        e = (self.modulation + e).chunk(6, dim=-2)  # [..., 1, D] each

        y = self.norm1(x) * (1 + e[1]) + e[0]  # [..., L, D]
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        x = x + (y * e[2])  # [..., L, D]

        if plucker_embedding is not None:
            camera_hidden_states = self.cam_injector_layer2(
                F.silu(self.cam_injector_layer1(plucker_embedding))
            )
            camera_hidden_states = camera_hidden_states + plucker_embedding
            camera_scale = self.cam_scale_layer(camera_hidden_states)
            camera_shift = self.cam_shift_layer(camera_hidden_states)
            x = (1.0 + camera_scale) * x + camera_shift

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e[4]) + e[3]  # [..., L, D]
        y = self.ffn(y)
        x = x + (y * e[5])  # [..., L, D]
        return x
