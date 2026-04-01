import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup

from flashsim.attention import BlockKVCache, RingAttention

from flashsim.model.video_dit.alpadreams.rope import apply_rope_freqs


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: Tensor) -> Tensor:
        return self.proj(image_embeds)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads * head_dim]
            e (Tensor): Modulation tensor with shape [batch_size, 1, n_heads * head_dim]

        Returns:
            Tensor: Output tensor with shape [batch_size, seq_len, n_heads * head_dim]
        """
        assert x.ndim == 3, "x is expected to be 3D tensor with shape [batch_size, seq_len, n_heads * head_dim]"
        assert e.ndim == 3, "e is expected to be 3D tensor with shape [batch_size, 1, n_heads * head_dim]"

        # TODO(ruilong): These can be fused into a normlinear layer.
        e = (self.modulation + e).chunk(2, dim=1)  # [B, 1, D] each
        x = self.norm(x) * (1 + e[1]) + e[0]  # [B, L, D]
        x = self.head(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention with KV cache and optional RoPE."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        eps: float = 1e-6,
    ) -> None:
        """Initialize a multi-head attention module.

        Args:
            query_dim: Feature dimension of query tokens and projected output.
            context_dim: Feature dimension of key/value tokens. Defaults to ``query_dim``.
            n_heads: Number of attention heads.
            head_dim: Per-head feature dimension. Inner dimension is ``n_heads * head_dim``.
        """
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.inner_dim = inner_dim
        self.eps = eps

        self.q = nn.Linear(query_dim, inner_dim)
        self.k = nn.Linear(context_dim, inner_dim)
        self.v = nn.Linear(context_dim, inner_dim)
        self.o = nn.Linear(inner_dim, query_dim)

        self.norm_q = nn.RMSNorm(inner_dim, eps=eps)
        self.norm_k = nn.RMSNorm(inner_dim, eps=eps)

        self.attn_op = RingAttention(qkv_format="bshd", backend="cudnn")

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Configure context-parallel process group for the underlying attention op."""
        self.attn_op.set_context_parallel_group(cp_group=cp_group)

    def is_context_parallel_enabled(self) -> bool:
        """Whether context parallelism is active for attention."""
        return self.attn_op.is_context_parallel_enabled()

    def context_parallel_size(self) -> int:
        """World size of the context-parallel group (1 if disabled)."""
        return self.attn_op.context_parallel_size()

    def _compute_or_update_kv_cache(
        self,
        context: Tensor,
        kv_cache: BlockKVCache | None = None,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Compute K/V from ``context`` and optionally merge into ``kv_cache``.

        Args:
            context: Tensor of shape [..., L, n * d] used to compute K/V.
            kv_cache: Existing cache to update, or None to allocate from this step.
            rope_freqs: RoPE frequencies, shape [L, 1, 1, d // 2].

        Returns:
            KV cache containing the merged keys and values.
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.norm_k(self.k(context)).reshape(batch_size, L, n, d)
        v = self.v(context).reshape(batch_size, L, n, d)
        if rope_freqs is not None:
            rope_freqs = torch.repeat_interleave(rope_freqs, repeats=2, dim=-1)
            k = apply_rope_freqs(k, rope_freqs, interleaved=True)

        if kv_cache is None:
            kv_cache = BlockKVCache.from_tensor(k, v, seq_dim=-3)
        else:
            kv_cache.update(k, v)
        return kv_cache

    def compute_kv(
        self,
        x: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Build a new KV cache from ``x``."""
        return self._compute_or_update_kv_cache(x, None, rope_freqs)

    def update_kv(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Append K/V computed from ``x`` into an existing ``kv_cache``."""
        return self._compute_or_update_kv_cache(x, kv_cache, rope_freqs)

    def apply_kv(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Run attention using ``x`` as queries and ``kv_cache`` as K/V source.

        Args:
            x: Query tensor of shape [..., L, n * d].
            kv_cache: KV cache for inference.
            rope_freqs: RoPE frequencies, shape [L, 1, 1, d // 2].

        Returns:
            Output tensor of shape [..., L, n * d] after projection.
        """
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        if rope_freqs is not None:
            rope_freqs = torch.repeat_interleave(rope_freqs, repeats=2, dim=-1)
            q = apply_rope_freqs(q, rope_freqs)

        cached_k = kv_cache.cached_k()
        cached_v = kv_cache.cached_v()

        out = self.attn_op(q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.o(out)

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
        update_kv_cache: bool = True,
    ) -> Tensor:
        """Optionally refresh cache from ``x`` and run attention.

        Args:
            x: Query tensor and, when updating, the source for new K/V ([..., L, n * d]).
            kv_cache: Cache read by attention; written when ``update_kv_cache`` is True.
            rope_freqs: Optional RoPE frequencies for Q and (when updating) K.
            update_kv_cache: If False, only run attention against the existing cache.

        Returns:
            Projected output tensor of shape [..., L, query_dim].
        """
        if update_kv_cache:
            kv_cache = self.update_kv(x, kv_cache, rope_freqs)
        return self.apply_kv(x, kv_cache, rope_freqs)


class SelfAttention(MultiHeadAttention):
    """Self-attention: queries and K/V are derived from the same ``x`` each step."""

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Initialize KV cache for streaming self-attention.

        Args:
            batch_size: Flattened batch size used by attention.
            chunk_size: Number of tokens appended per update step.
            window_size: Rolling-window capacity in tokens.
            sink_size: Sink-token capacity retained permanently.
            device: Device for cache tensors.
            dtype: Data type for cache tensors.

        Returns:
            An initialized ``BlockKVCache``.
        """
        total_size = sink_size + window_size
        return BlockKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Same as base ``forward`` with ``update_kv_cache=True``."""
        return super().forward(x, kv_cache, rope_freqs=rope_freqs, update_kv_cache=True)


@dataclass
class CrossAttnCache:
    """Cache container for cross-attention."""
    
    text: BlockKVCache
    img: BlockKVCache | None = None  # only used for I2V


class CrossAttention(MultiHeadAttention):
    """Cross-attention: K/V live only in ``kv_cache``; ``forward`` does not refresh them."""

    def __init__(self, i2v: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.i2v = i2v
        if i2v:
            self.k_img = nn.Linear(self.context_dim, self.inner_dim)
            self.v_img = nn.Linear(self.context_dim, self.inner_dim)
            self.norm_k_img = nn.RMSNorm(self.inner_dim, eps=self.eps)
            self.attn_op_image = RingAttention(qkv_format="bshd", backend="cudnn")

    def compute_kv_image(self, context: Tensor) -> BlockKVCache:
        """Compute K/V from image ``context``.

        Args:
            context: Tensor of shape [..., L, n * d] used to compute K/V.

        Returns:
            KV cache containing the merged keys and values.
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.norm_k_img(self.k_img(context)).reshape(batch_size, L, n, d)
        v = self.v_img(context).reshape(batch_size, L, n, d)
        return BlockKVCache.from_tensor(k, v, seq_dim=-3)

    def initialize_cache(
        self,
        context_text: Tensor,  # [B, L, D]
        context_img: Tensor | None = None,  # [B, L, D]
    ) -> CrossAttnCache:
        """Initialize cross-attention cache from the provided context."""
        text_cache = self.compute_kv(context_text)
        if self.i2v:
            img_cache = self.compute_kv_image(context_img)
        else:
            img_cache = None
        return CrossAttnCache(text=text_cache, img=img_cache)

    def forward(
        self,
        x: Tensor,
        kv_cache: CrossAttnCache,
    ) -> Tensor:
        """Attend with queries from ``x``"""
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.norm_q(self.q(x)).reshape(batch_size, L, n, d)
        out = self.attn_op(q, kv_cache.text.cached_k(), kv_cache.text.cached_v())
        if self.i2v:
            assert kv_cache.img is not None, "kv_cache_img is expected to be provided for I2V cross-attention"
            out_img = self.attn_op_image(q, kv_cache.img.cached_k(), kv_cache.img.cached_v())
            out = out + out_img
        out = out.reshape(batch_shape + (L, n * d))
        return self.o(out)

@dataclass
class BlockCache:
    """Per-block cache container for self-attention and cross-attention."""

    self_attn: BlockKVCache
    cross_attn: BlockKVCache

    def before_update(self, chunk_idx: int) -> None:
        """Run cache pre-update hook for the current chunk."""
        self.self_attn.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Run cache post-update hook for the current chunk."""
        self.self_attn.after_update(chunk_idx)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=True,
        eps=1e-6,
        i2v=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = SelfAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            eps=eps,
        )
        self.norm3 = nn.LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = CrossAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            i2v=i2v,
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def initialize_cache(
        self, 
        # self-attention
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross-attention
        context_text: Tensor,  # [B, L, D]
        context_img: Tensor | None = None,  # [B, L, D]
    ) -> BlockCache:
        """Initialize per-branch caches for this transformer block."""
        batch_size = context_text.shape[0]
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
        self.self_attn.set_context_parallel_group(cp_group)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        block_kv_cache: BlockCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads * head_dim]
            e (Tensor): Modulation tensor with shape [batch_size, 6, n_heads * head_dim]
            block_kv_cache (BlockCache): KV cache for the attention block
            rope_freqs (Tensor): RoPE frequencies with shape [seq_len, 1, 1, head_dim // 2]
        Returns:
            Tensor: Output tensor with shape [batch_size, seq_len, n_heads * head_dim]
        """
        e = (self.modulation + e).chunk(6, dim=1)  # [B, 1, D] each

        y = self.norm1(x) * (1 + e[1]) + e[0]  # [B, L, D]
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=block_kv_cache.self_attn,
        )
        x = x + (y * e[2])  # [B, L, D]

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=block_kv_cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e[4]) + e[3]  # [B, L, D]
        y = self.ffn(y)
        x = x + (y * e[5])  # [B, L, D]
        return x
