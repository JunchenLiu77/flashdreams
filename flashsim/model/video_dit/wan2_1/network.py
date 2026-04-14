from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

from flashsim.distributed.context_parallel import cat_outputs_cp, split_inputs_cp
from flashsim.configs import InstantiateConfig

from flashsim.model.video_dit.wan2_1.modules import (
    BlockCache,
    Block,
    Head,
    MLPProj,
    sinusoidal_embedding_1d,
)


@dataclass
class WanDiTNetworkCache:
    """Cache container for all transformer blocks."""

    block_caches: list[BlockCache]

    def __getitem__(self, index: int) -> BlockCache:
        """Get cache for a specific block."""
        return self.block_caches[index]

    def before_update(self, chunk_idx: int) -> None:
        """Run pre-update hooks for all block caches."""
        for block_cache in self.block_caches:
            block_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        """Run post-update hooks for all block caches."""
        for block_cache in self.block_caches:
            block_cache.after_update(chunk_idx)


@dataclass
class WanDiTNetworkConfig(InstantiateConfig["WanDiTNetwork"]):
    _target: type["WanDiTNetwork"] = field(default_factory=lambda: WanDiTNetwork)

    model_type: Literal["t2v", "i2v"] = "t2v"
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 16
    dim: int = 1536
    ffn_dim: int = 8960
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 12
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    concat_padding_mask: bool = False
    additional_concat_ch: int = 0
    patch_embedding_type: Literal["linear", "conv3d"] = "linear"


@dataclass
class WanDiTNetwork1pt3BConfig(WanDiTNetworkConfig):
    """Configuration for the 1.3B Wan DiT network."""

    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30


@dataclass
class WanDiTNetwork14BConfig(WanDiTNetworkConfig):
    """Configuration for the 14B Wan DiT network."""

    dim: int = 5120
    ffn_dim: int = 13824
    num_heads: int = 40
    num_layers: int = 40


class WanDiTNetwork(nn.Module):
    """WAN diffusion backbone for text-to-video and image-to-video."""

    def __init__(self, config: WanDiTNetworkConfig):
        """Initialize WAN DiT backbone.

        Args:
            model_type: Model variant. Supported values: ``"t2v"``, ``"i2v"``.
            patch_size: 3D patch size ``(t_patch, h_patch, w_patch)``.
            text_len: Fixed maximum text length.
            in_dim: Input latent channels.
            dim: Transformer hidden dimension.
            ffn_dim: Feed-forward hidden dimension.
            freq_dim: Sinusoidal timestep embedding dimension.
            text_dim: Input text embedding dimension.
            out_dim: Output latent channels.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks.
            cross_attn_norm: Whether to apply normalization before cross-attention.
            eps: Epsilon for normalization layers.
            concat_padding_mask: Whether one mask channel is concatenated into input.
            additional_concat_ch: Extra conditioning channels (e.g. HDMap).
        """

        super().__init__()

        assert config.model_type in ["t2v", "i2v"], "model_type must be 't2v' or 'i2v'"
        self.model_type = config.model_type

        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.dim = config.dim
        self.ffn_dim = config.ffn_dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.out_dim = config.out_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.cross_attn_norm = config.cross_attn_norm
        self.eps = config.eps
        self.concat_padding_mask = config.concat_padding_mask
        self.additional_concat_ch = config.additional_concat_ch
        self.patch_embedding_type = config.patch_embedding_type

        # Embedding layers
        in_dim = config.in_dim + 1 if self.concat_padding_mask else config.in_dim
        if config.patch_embedding_type == "linear":
            self.patch_embedding = nn.Linear(
                in_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                self.dim,
            )
        elif config.patch_embedding_type == "conv3d":
            self.patch_embedding = nn.Conv3d(
                in_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        else:
            raise ValueError(
                f"Invalid patch embedding type: {config.patch_embedding_type}"
            )
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dim, self.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6)
        )
        if self.model_type == "i2v":
            self.img_emb = MLPProj(1280, self.dim)
        if self.additional_concat_ch > 0:
            self.additional_patch_embedding = nn.Linear(
                self.additional_concat_ch
                * self.patch_size[0]
                * self.patch_size[1]
                * self.patch_size[2],
                self.dim,
            )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.cross_attn_norm,
                    self.eps,
                    i2v=(self.model_type == "i2v"),
                )
                for _ in range(self.num_layers)
            ]
        )

        # Final projection head
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)

        self._is_shuffle_op_fused = False
        self._parameters_updated_after_loading_checkpoint = False

    def set_context_parallel_group(self, cp_group: ProcessGroup | None = None) -> None:
        """Set context-parallel process group for all blocks.

        This must be called before ``initialize_cache`` when CP is used.
        """
        for block in self.blocks:
            block.set_context_parallel_group(cp_group)

    def patchify_and_maybe_split_cp(
        self,
        x: Tensor,  # [..., T, C, H, W]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        r"""
        Patchify the input tensor and maybe split it along cp_dim if a process group is provided.

        The patchify pattern is:
            "... (t kt) c (h kh) (w kw) -> ... (t h w) (c kt kh kw)",

        Returns:
            Tensor: The patched tensor with shape [..., L, D], where L = T * H * W / (kt * kh * kw)
        """
        assert x.ndim == 6, f"x must be a 6D tensor, but got shape {x.shape}"

        x = rearrange(
            x,
            "... (t kt) c (h kh) (w kw) -> ... (t h w) (c kt kh kw)",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
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
        x: Tensor,  # [..., L, D]
        process_groups: list[ProcessGroup | None] | None = None,
        cp_dims: list[int | None] | None = None,
    ) -> Tensor:
        r"""
        Unpatchify the input tensor and maybe gather it along cp_dim if a process group is provided.

        The unpatchify pattern is:
            "... (t h w) (c kt kh kw) -> ... (t kt) c (h kh) (w kw)",

        Returns:
            Tensor: The unpatched tensor with shape [..., T, C, H, W]
        """
        assert x.ndim >= 2, f"x must be a 2D or higher tensor, but got shape {x.shape}"

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
            "... (t h w) (c kt kh kw) -> ... (t kt) c (h kh) (w kw)",
            h=pH,
            w=pW,
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
        )
        return x  # [..., T, C, H, W]

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        text_embeddings: Tensor,
        img_embeddings: Tensor | None = None,
    ) -> WanDiTNetworkCache:
        """Initialize block caches from text/image context embeddings.

        Args:
            chunk_size: Number of tokens appended per self-attention update.
            window_size: Rolling-window size in tokens for self-attention cache.
            sink_size: Sink-token capacity preserved across updates.
            text_embeddings: Text embeddings. UMT5 has shape [..., 512, 4096].
            img_embeddings: Optional image embeddings for I2V. CLIP has shape [..., 256, 1280].

        Returns:
            ``WanDiTNetworkCache`` containing per-block caches.
        """
        assert text_embeddings.shape[-2] == self.text_len
        context_text = self.text_embedding(text_embeddings)
        if self.model_type == "i2v":
            assert img_embeddings is not None, (
                "img_embeddings is required when model_type='i2v'"
            )
            context_img = self.img_emb(img_embeddings)
        else:
            context_img = None

        return WanDiTNetworkCache(
            block_caches=[
                block.initialize_cache(
                    chunk_size, window_size, sink_size, context_text, context_img
                )
                for block in self.blocks
            ],
        )

    def update_parameters_after_loading_checkpoint(self) -> None:
        # This function should be called after loading the checkpoint, to fuse some operations in the model
        # weights to reduce computation during inference.
        if self._parameters_updated_after_loading_checkpoint:
            return

        self._fuse_shuffle_op_into_last_layer()
        self._parameters_updated_after_loading_checkpoint = True

    def _fuse_shuffle_op_into_last_layer(self) -> None:
        """
        In the WAN model, the patchify operation is
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
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
            c=self.out_dim,
        )
        ```
        """
        if self._is_shuffle_op_fused:
            return

        self.head.head.weight.data = rearrange(
            self.head.head.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
            c=self.out_dim,
        ).contiguous()
        if self.head.head.bias is not None:
            self.head.head.bias.data = rearrange(
                self.head.head.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.patch_size[0],
                kh=self.patch_size[1],
                kw=self.patch_size[2],
                c=self.out_dim,
            ).contiguous()

        self.is_shuffle_op_fused = True

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        hdmap_condition: Tensor | None = None,
        eager_mode: bool = True,
    ) -> Tensor:
        """Run one denoising forward pass.

        Args:
            x: Input tokens of shape [..., L, D_in] after patchify.
                The layout is assumed to be
                "... (t h w) (c kt kh kw)".
            timesteps: Diffusion timesteps of shape [...].
            cache: Per-block KV caches.
            rope_freqs: RoPE frequencies of shape [L, 1, 1, head_dim // 2] after CP.
            current_chunk_idx: Current chunk index for streaming cache update.
            hdmap_condition: Optional HDMap tensor of shape [..., L, D_hdmap] after patchify.
            eager_mode: If True, run cache before/after update hooks.

        Returns:
            Tensor of shape [..., L, prod(patch_size) * out_dim].
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )
        batch_shape = x.shape[:-2]

        # Patch embedding
        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)  # (..., L, D)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(
                self.dim, -1
            )  # [D, in_dim * kt * kh * kw]
            _bias = self.patch_embedding.bias  # [D] or None
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        # Optional HDMap embedding
        if self.additional_concat_ch > 0:
            assert hdmap_condition is not None, (
                "hdmap is expected to be provided for additional concat channels"
            )
            additional_x = self.additional_patch_embedding(hdmap_condition)
            x = x + additional_x  # (..., L, D)

        # Timestep embedding and modulation projection
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )  # [..., D]
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))  # [..., 6, D]

        # Transformer blocks
        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            x = block(
                x=x,
                e=torch.broadcast_to(e0, batch_shape + e0.shape[-2:]),
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)

        # Final head
        x = self.head(
            x, torch.broadcast_to(e, batch_shape + (1, e.shape[-1]))
        )  # (..., L, D)
        return x


def test_basic(i2v: bool = False, use_hdmap: bool = False) -> None:
    """Quick local smoke test for WAN network forward pass."""
    torch.manual_seed(42)
    # 14B model
    device = "cuda"
    dtype = torch.bfloat16

    additional_concat_ch = 0
    if i2v:
        model_type = "i2v"
        in_dim = 16 + 20  # 16 is noise, 20 is image conditioning
        if use_hdmap:
            additional_concat_ch = 16
    else:
        model_type = "t2v"
        in_dim = 16

    T, H, W = 3, 720 // 8, 1280 // 8
    num_tokens_per_frame = H // 2 * W // 2
    num_tokens_per_chunk = T * num_tokens_per_frame

    network = (
        WanDiTNetworkConfig(
            model_type=model_type,
            dim=5120,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=in_dim,
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
            additional_concat_ch=additional_concat_ch,
        )
        .setup()
        .to(device=device, dtype=dtype)
    )
    # torch.save(network.state_dict(), "outputs/wan2_1_network.pth")
    network.load_state_dict(torch.load("outputs/wan2_1_network.pth"))

    torch.manual_seed(42)
    data = torch.randn(1, in_dim, T, H, W, device=device, dtype=dtype)
    x = rearrange(
        data,
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
        kt=network.patch_size[0],
        kh=network.patch_size[1],
        kw=network.patch_size[2],
    )
    timesteps = torch.randn(1, device=device, dtype=dtype)
    rope_freqs = torch.randn(
        num_tokens_per_chunk, 1, 1, 64, device=device, dtype=torch.float32
    )
    _camera = torch.randn(1, num_tokens_per_chunk, 1536, device=device, dtype=dtype)
    if use_hdmap:
        hdmap_condition = torch.randn(
            1, additional_concat_ch, T, H, W, device=device, dtype=dtype
        )
        hdmap_condition = rearrange(
            hdmap_condition,
            "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
            kt=network.patch_size[0],
            kh=network.patch_size[1],
            kw=network.patch_size[2],
        )

    else:
        hdmap_condition = None

    network.initialize_context_parallel()
    network.update_parameters_after_loading_checkpoint()

    network_cache = network.initialize_cache(
        chunk_size=num_tokens_per_chunk,
        window_size=21 * num_tokens_per_frame,
        sink_size=3 * num_tokens_per_frame,
        text_embeddings=torch.randn(1, 512, 4096, device=device, dtype=dtype),
        img_embeddings=torch.randn(1, 256, 1280, device=device, dtype=dtype),
    )

    @torch.no_grad()
    def _run():
        output = network(
            x,
            timesteps,
            network_cache,
            rope_freqs=rope_freqs,
            current_chunk_idx=0,
            hdmap_condition=hdmap_condition,
        )
        return output

    output = _run()

    print(
        "i2v:",
        i2v,
        "use_hdmap:",
        use_hdmap,
        "x.shape:",
        x.shape,
        "output.shape:",
        output.shape,
        "output.sum():",
        output.sum(),
    )
    # i2v: True use_hdmap: False x.shape: torch.Size([1, 10800, 144]) output.shape: torch.Size([1, 10800, 64]) output.sum(): tensor(10176., device='cuda:0', dtype=torch.bfloat16)


# torchrun --nproc_per_node=1 flashsim/model/video_dit/wan2_1/network.py
if __name__ == "__main__":
    test_basic(i2v=True)
    # test_basic(i2v=False)
    # test_basic(i2v=True, use_hdmap=True)
