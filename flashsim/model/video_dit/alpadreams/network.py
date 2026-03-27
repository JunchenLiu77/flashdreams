import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.distributed import ProcessGroup

from flashsim.attention import BlockKVCache, RingAttention
from flashsim.distributed.context_parallel import cat_outputs_cp, split_inputs_cp

from .modules import (
    FinalLayer,
    GPT2FeedForward,
    PatchEmbed,
    TimestepEmbedding,
    Timesteps,
)
from .rope import apply_rope_freqs


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional KV cache, RoPE, and ring attention backend."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        """
        Args:
            query_dim: Last-dim size of queries (and output).
            context_dim: Last-dim size of key/value inputs; defaults to ``query_dim``.
            n_heads: Number of attention heads.
            head_dim: Dimension per head; inner dim is ``n_heads * head_dim``.
        """
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

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
        """
        Update the KV cache with new keys and values.

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

        k = self.k_norm(self.k_proj(context).reshape(batch_size, L, n, d))
        v = self.v_proj(context).reshape(batch_size, L, n, d)
        if rope_freqs is not None:
            k = apply_rope_freqs(k, rope_freqs)

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
        return self._compute_or_update_kv_cache(x, None, rope_freqs)

    def update_kv(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        return self._compute_or_update_kv_cache(x, kv_cache, rope_freqs)

    def apply_kv(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """
        Attention over queries x using cached K/V (read-only on the cache).

        Args:
            x: Query tensor of shape [..., L, n * d].
            rope_freqs: RoPE frequencies, shape [L, 1, 1, d // 2].
            kv_cache: KV cache for inference.

        Returns:
            Output tensor of shape [..., L, n * d] after projection.
        """
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        assert n * d == D, "n * d must be equal to D"

        q = self.q_norm(self.q_proj(x).reshape(batch_size, L, n, d))
        if rope_freqs is not None:
            q = apply_rope_freqs(q, rope_freqs)

        cached_k = kv_cache.cached_k()
        cached_v = kv_cache.cached_v()

        out = self.attn_op(q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.output_proj(out)

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
        update_kv_cache: bool = True,
    ) -> Tensor:
        """
        If ``update_kv_cache``, update ``kv_cache`` from ``x`` (K/V source), then attend.

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
        """
        Initialize the cache for the attention.
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


class CrossAttention(MultiHeadAttention):
    """Cross-attention: K/V live only in ``kv_cache``; ``forward`` does not refresh them."""

    def initialize_cache(
        self,
        context: Tensor, # [B, V, L, D]
    ) -> BlockKVCache:
        """
        Initialize the cache for the attention.
        """
        cache = self.compute_kv(context)
        return cache

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
    ) -> Tensor:
        """Attend with queries from ``x``; populate or roll ``kv_cache`` outside this call."""
        return super().forward(x, kv_cache, rope_freqs=None, update_kv_cache=False)


@dataclass
class CosmosBlockCache:
    self_attn: BlockKVCache
    cross_attn: BlockKVCache

    def before_update(self, chunk_idx: int) -> None:
        self.self_attn.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        self.self_attn.after_update(chunk_idx)


class CosmosBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and MLP.

    Uses AdaLN modulation for timestep conditioning following the Cosmos architecture.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        enable_cross_view_attn: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.enable_cross_view_attn = enable_cross_view_attn

        # Self-attention
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(
            query_dim=x_dim,
            context_dim=None,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
        )

        # Cross-attention
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            query_dim=x_dim,
            context_dim=context_dim,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
        )

        # MLP
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        # AdaLN modulation
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        if enable_cross_view_attn:
            # no modulation so we set elementwise_affine=True
            self.layer_norm_cross_view_attn = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)
            # dense cross view attention
            self.cross_view_attn = CrossAttention(
                query_dim=x_dim,
                context_dim=x_dim,
                n_heads=num_heads,
                head_dim=x_dim // num_heads,
            )

    def set_context_parallel_group(
        self, self_attn_group: ProcessGroup | None, cross_view_attn_group: ProcessGroup | None = None
    ):
        """Set hierarchical CP groups for self-attention and cross-view attention.

        Args:
            self_attn_group: Group for ranks processing same view (for T gathering in self-attention)
            cross_view_attn_group: Group for ranks at same T slice (for V gathering in cross-view)
        """
        # Self-attention uses self_attn_group (for T gathering)
        self.self_attn.set_context_parallel_group(cp_group=self_attn_group)
        # Cross-view attention uses cross_view_attn_group (for V gathering)
        if self.enable_cross_view_attn:
            self.cross_view_attn.set_context_parallel_group(cp_group=cross_view_attn_group)

    def initialize_cache(
        self,
        # self attn
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross attn
        context: Tensor, # [B, V, L, D]
    ) -> CosmosBlockCache:
        """
        Initialize the cache for the block.
        """
        device = context.device
        dtype = context.dtype
        batch_size = context.shape[0]
        num_views = context.shape[1]
        self_attn_batch_size = batch_size * num_views
        return CosmosBlockCache(
            self_attn=self.self_attn.initialize_cache(
                self_attn_batch_size, chunk_size, window_size, sink_size, device=device, dtype=dtype,
            ),
            cross_attn=self.cross_attn.initialize_cache(context),
        )

    def forward(
        self,
        x: Tensor,  # [B, V, T, HW, D]
        emb: Tensor,
        cache: CosmosBlockCache,
        rope_freqs: Tensor,  # [L, 1, 1, D]
        adaln_lora: Tensor | None = None,
        view_embedding_proj: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through the block.

        Args:
            x: Input tensor [B, V, T, HW, D]
            emb: Time embedding [B, D]
            cache: CosmosBlockCache
            rope_freqs: RoPE cosine and sine embeddings [L, 1, 1, D]
            adaln_lora: AdaLN LoRA embeddings [B, 3D]
            view_embedding_proj: View embedding projection [B, V, 9D]
        """
        B, V, T, HW, D = x.shape

        # reshape embeddings to be broadcastable with x.
        emb = emb.reshape(B, 1, 1, 1, D)

        # Compute AdaLN modulation
        if self.use_adaln_lora:
            assert adaln_lora is not None, "adaln_lora is required when use_adaln_lora is True"
            adaln_lora = adaln_lora.reshape(B, 1, 1, 1, 3 * D)
            shift_self, scale_self, gate_self = (self.adaln_modulation_self_attn(emb) + adaln_lora).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = (self.adaln_modulation_cross_attn(emb) + adaln_lora).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb) + adaln_lora).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb).chunk(3, dim=-1)

        if self.enable_cross_view_attn:
            assert view_embedding_proj is not None
            (
                view_shift_self,
                view_scale_self,
                view_gate_self,
                view_shift_cross,
                view_scale_cross,
                view_gate_cross,
                view_shift_mlp,
                view_scale_mlp,
                view_gate_mlp,
            ) = view_embedding_proj.chunk(9, dim=-1)

            def expand_view_mod(v_mod: Tensor) -> Tensor:
                return v_mod.reshape(B, V, 1, 1, D)

            shift_self = shift_self + expand_view_mod(view_shift_self)
            scale_self = scale_self + expand_view_mod(view_scale_self)
            gate_self = gate_self + expand_view_mod(view_gate_self)

            shift_cross = shift_cross + expand_view_mod(view_shift_cross)
            scale_cross = scale_cross + expand_view_mod(view_scale_cross)
            gate_cross = gate_cross + expand_view_mod(view_gate_cross)

            shift_mlp = shift_mlp + expand_view_mod(view_shift_mlp)
            scale_mlp = scale_mlp + expand_view_mod(view_scale_mlp)
            gate_mlp = gate_mlp + expand_view_mod(view_gate_mlp)

        # Self-attention (API aligned with Attention.forward)
        normed_x = self.layer_norm_self_attn(x) * (1 + scale_self) + shift_self
        attn_out = self.self_attn(
            normed_x.reshape(B, V, -1, D),
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        ).reshape_as(normed_x)
        x = x + gate_self * attn_out

        # Cross-view attention: dense
        if self.enable_cross_view_attn:
            normed_x_cv = self.layer_norm_cross_view_attn(x)
            x = rearrange(normed_x_cv, f"b v t hw d -> b t v hw d")
            if self.cross_view_attn.cp_enabled:
                # Note: we cross view attention is CP enabled, we assume multi-view is split across GPU
                # ranks IN ORDER. E.g., for 4 views on 2 GPUs, we assume the groups are [0, 1] views and [2, 3] views.
                if V == 1:
                    # When cross attention is CP enabled, and the CP size is equal to the number of views,
                    # then each gpu processes exactly one view. Since attention will gather
                    # all KV from all gpus, each gpu effectively only need to process KV for its own view.
                    x_context = x
                    # effectively same as the following, but since V=1 it results in the same tensor.
                    # x_context = repeat(x, f"b t v hw d -> b t v2 (v hw) d", v2=V)
                else:
                    # When CP size is less than the number of views, e.g., for 4 views on 2 GPUs,
                    # each gpu processes multiple views. We can still rely on attention to gather
                    # all KV from all gpus, but in this case KV on each gpu should cover 4/2 = 2 views.
                    x_context = repeat(x, f"b t v hw d -> b t v2 (v hw) d", v2=V)
            else:
                # When cross attention is not CP enabled, we need to repeat the context
                # to match the number of views. such that the attention will be computed
                # across all views.
                x_context = repeat(x, f"b t v hw d -> b t v2 (v hw) d", v2=V)
            cross_view_attn_kv_cache = self.cross_view_attn.compute_kv(x_context)
            cv_out = self.cross_view_attn(x, kv_cache=cross_view_attn_kv_cache)
            cv_out = rearrange(cv_out, f"b t v hw d -> b v t hw d")
            x = x + cv_out

        # Cross-attention
        normed_x = self.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
        cross_out = self.cross_attn(normed_x.reshape(B, V, -1, D), kv_cache=cache.cross_attn).reshape_as(normed_x)
        x = x + gate_cross * cross_out

        # MLP
        normed_x = self.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x = x + gate_mlp * mlp_out

        return x


@dataclass
class CosmosDiTCache:
    block_caches: list[CosmosBlockCache]

    def __getitem__(self, index: int) -> CosmosBlockCache:
        return self.block_caches[index]

    def before_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.after_update(chunk_idx)


class CosmosDiT(nn.Module):
    """
    DiT model for video generation with block-causal attention and KV-caching.

    Combines the Cosmos DiT architecture with causal attention masking for
    autoregressive video generation.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        concat_padding_mask: bool = True,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        use_crossattn_projection: bool = True,
        crossattn_proj_in_channels: int = 100352,
        crossattn_emb_channels: int = 1024,
        timestep_scale: float = 0.001,
        # hdmap conditioning
        additional_concat_ch: int = 0,
        # multiview
        enable_cross_view_attn: bool = False,
        view_condition_dim: int = 16,
        n_cameras_emb: int = 7,
    ):
        super().__init__()
        self.timestep_scale = timestep_scale
        # add 1 for the condition mask
        self.in_channels = in_channels + 1
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask

        # Positional embedding settings
        self.additional_concat_ch = additional_concat_ch

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels
        self.enable_cross_view_attn = enable_cross_view_attn
        # Build embeddings
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=self.in_channels + 1 if self.concat_padding_mask else self.in_channels,
            out_channels=self.model_channels,
        )
        # HDMap conditioning
        if self.additional_concat_ch > 0:
            self.additional_patch_embedding = PatchEmbed(
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                in_channels=self.additional_concat_ch,
                out_channels=self.model_channels,
            )

        # Time embeddings
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.t_embedding_norm = nn.RMSNorm(model_channels, eps=1e-6)

        # Transformer blocks (API aligned with Block from minimal_v4_dit)
        self.blocks = nn.ModuleList(
            [
                CosmosBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    enable_cross_view_attn=enable_cross_view_attn,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        if enable_cross_view_attn:
            self.adaln_view_embedder = nn.Embedding(n_cameras_emb, model_channels)
            self.adaln_view_proj = nn.Linear(model_channels, model_channels * 9)
        else:
            self.adaln_view_embedder = None
            self.adaln_view_proj = None

        self._is_shuffle_op_fused = False
        self._is_padding_mask_fused = False
        self._parameters_updated_after_loading_checkpoint = False

    def set_context_parallel_group(
        self, self_attn_group: ProcessGroup | None, cross_view_attn_group: ProcessGroup | None = None
    ) -> None:
        for block in self.blocks:
            block.set_context_parallel_group(self_attn_group, cross_view_attn_group)

    def _fuse_shuffle_op_into_last_layer(self):
        """
        In the Cosmos model, the patchify operation is
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",

        while the unpatchify operation is
        "b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)"

        This is likely a bug in the Cosmos model where the last dimension is shuffled after the network.

        To fix this, we could fuse this shuffle op into the last linear layer,
        so that we do not have to do this shuffle op explicitly before returning the result.

        Calling this function to modify the last layer in place, is equivalent to the following code
        after the last layer:
        ```python
        x = rearrange(
            x,
            "... (kt kh kw c) -> ... (c kt kh kw)",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
            c=self.out_channels,
        )
        ```
        """
        if self._is_shuffle_op_fused:
            return

        self.final_layer.linear.weight.data = rearrange(
            self.final_layer.linear.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
            c=self.out_channels,
        ).contiguous()
        if self.final_layer.linear.bias is not None:
            self.final_layer.linear.bias.data = rearrange(
                self.final_layer.linear.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.patch_temporal,
                kh=self.patch_spatial,
                kw=self.patch_spatial,
                c=self.out_channels,
            ).contiguous()

        self._is_shuffle_op_fused = True
        return

    def _fuse_padding_mask_into_patch_embed(self) -> None:
        """
        Fuse the padding mask into the patch embedder in place.

        If `self.concat_padding_mask` is True, during training we are concatenating a
        padding_mask with shape [B, 1, T, H, W] to the input x_B_C_T_H_W on the C dimension,
        before passing it into the self.x_embedder. This is to work with data with different
        spatial resolutions during training, where `1` indicates padded regions. During
        inference, the padding_mask is always 0. So here we could simply remove the corresponding
        channels in the x_embedder in place

        Calling this function to modify the patch embedder in place, is equivalent to the following code
        before passing the input into the patch embedder:
        ```python
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask], dim=1)
        ```
        """
        if not self.concat_padding_mask:
            return

        if self._is_padding_mask_fused:
            return

        self.x_embedder.in_channels -= 1
        in_channels_to_keep = self.x_embedder.get_linear_in_channels()
        self.x_embedder.proj[1].weight.data = self.x_embedder.proj[1].weight.data[:, :in_channels_to_keep].contiguous()
        if self.x_embedder.proj[1].bias is not None:
            self.x_embedder.proj[1].bias.data = self.x_embedder.proj[1].bias.data[:in_channels_to_keep].contiguous()

        self._is_padding_mask_fused = True
        return

    def update_parameters_after_loading_checkpoint(self) -> None:
        # This function should be called after loading the checkpoint, to fuse some operations in the model
        # weights to reduce computation during inference.
        if self._parameters_updated_after_loading_checkpoint:
            return

        self._fuse_padding_mask_into_patch_embed()
        self._fuse_shuffle_op_into_last_layer()
        self._parameters_updated_after_loading_checkpoint = True

    def patchify_and_maybe_split_cp(
        self,
        x: Tensor,  # [B, V, C, T, H, W]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        r"""
        Patchify the input tensor and maybe split it along cp_dim if a process group is provided.

        The patchify pattern is:
            "b v c (t kt) (h kh) (w kw) -> b v t (h w) (c kt kh kw)",

        Returns:
            Tensor: The patched tensor with shape [B, V, T, HW, D]
        """
        assert x.ndim == 6, f"x must be a 6D tensor, but got shape {x.shape}"

        x = rearrange(
            x,
            f"... v c (t kt) (h kh) (w kw) -> ... v t (h w) (c kt kh kw)",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
        )

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided"
                "and the number of dimensions must match the number of process groups"
            )
            for cp_dim, process_group in zip(cp_dims, process_groups):
                if process_group is not None:
                    assert cp_dim is not None, (
                        "Context parallel dimension must be provided if process group is provided"
                    )
                    x = split_inputs_cp(x, seq_dim=cp_dim, cp_group=process_group)
        return x

    def unpatchify_and_maybe_gather_cp(
        self,
        pH: int,
        pW: int,
        x: Tensor,  # [B, V, T, HW, D]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        r"""
        Unpatchify the input tensor and maybe gather it along cp_dim if a process group is provided.

        The unpatchify pattern is:
            "b v t (h w) (c kt kh kw) -> b v c (t kt) (h kh) (w kw)",

        Returns:
            Tensor: The unpatched tensor with shape [B, V, C, T, H, W]
        """
        assert x.ndim == 5, f"x must be a 5D tensor, but got shape {x.shape}"

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided"
                "and the number of dimensions must match the number of process groups"
            )
            for cp_dim, process_group in zip(cp_dims, process_groups):
                if process_group is not None:
                    assert cp_dim is not None, (
                        "Context parallel dimension must be provided if process group is provided"
                    )
                    x = cat_outputs_cp(x, seq_dim=cp_dim, cp_group=process_group)

        x = rearrange(
            x,
            f"b v t (h w) (c kt kh kw) -> b v c (t kt) (h kh) (w kw)",
            h=pH,
            w=pW,
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
        )
        return x  # [B, V, C, T, H, W]

    def initialize_cache(
        self,
        # self attn
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross attn
        context: Tensor,
    ) -> CosmosDiTCache:
        """
        Initialize the cache for the DiT.
        """
        if self.use_crossattn_projection:
            context = self.crossattn_proj(context)

        return CosmosDiTCache(
            block_caches=[
                block.initialize_cache(chunk_size, window_size, sink_size, context)
                for block in self.blocks
            ],
        )

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        rope_freqs: Tensor,  # [L, 1, 1, D]
        cache: CosmosDiTCache,
        condition_video_input_mask: Tensor,
        current_chunk_idx: int = 0,
        hdmap_condition: Tensor | None = None,
        view_indices: Tensor | None = None,
        eager_mode: bool = True,
    ) -> Tensor:
        """
        Forward pass dispatching to training or inference mode.

        Args:
            x: Input video tensor [B, V, T, HW, D] after patchify
            timesteps: Timesteps [1] or [B]
            rope_freqs: RoPE cosine and sine embeddings [T*HW, 1, 1, D]
            cache: CosmosDiTCache
            condition_video_input_mask: Condition video input mask [B, V, T, HW, D] after patchify
            current_chunk_idx: Current chunk index in autoregressive inference
            hdmap_condition: HDMap tensor [B, V, T, HW, D]
            view_indices: View indices [B, V]
            eager_mode: Whether to run in eager mode (True) or not (False)
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )

        assert timesteps.ndim == 1
        timesteps = timesteps * self.timestep_scale

        # Patch embedding
        x = torch.cat([x, condition_video_input_mask], dim=-1)
        x = self.x_embedder(x)

        if self.additional_concat_ch > 0:
            assert hdmap_condition is not None, "hdmap is expected to be provided for additional concat channels"
            additional_x = self.additional_patch_embedding(hdmap_condition)
            x = x + additional_x

        # Time embedding
        t_emb, adaln_lora = self.t_embedder(timesteps)
        t_emb = self.t_embedding_norm(t_emb)

        # AdaLN view modulation if enabled
        if view_indices is not None:
            assert self.adaln_view_embedder is not None and self.adaln_view_proj is not None, (
                "adaln_view_embedder and adaln_view_proj must be provided if view_indices_B_V is provided"
            )
            view_emb = self.adaln_view_embedder(view_indices)  # [B, V, D]
            view_embedding_proj = self.adaln_view_proj(view_emb)  # [B, V, 9D]
        else:
            view_embedding_proj = None

        # Note: If not in eager mode, we should call `before_update` and `after_update` MANUALLY outside the network.
        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            x = block(
                x=x,
                emb=t_emb,
                rope_freqs=rope_freqs,
                adaln_lora=adaln_lora,
                cache=cache[block_idx],
                view_embedding_proj=view_embedding_proj,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        # Final layer
        x = self.final_layer(x, t_emb, adaln_lora)
        return x