from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

from flashsim.model.video_dit.wan2_1.modules import (
    BlockCache,
    Block,
    Head,
    MLPProj,
    sinusoidal_embedding_1d,
)


@dataclass
class WanDiTNetworkCache:
    block_caches: list[BlockCache]

    def __getitem__(self, index: int) -> BlockCache:
        return self.block_caches[index]

    def before_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.before_update(chunk_idx)

    def after_update(self, chunk_idx: int) -> None:
        for block_cache in self.block_caches:
            block_cache.after_update(chunk_idx)


class WanDiTNetwork(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        cross_attn_norm=True,
        eps=1e-6,
        concat_padding_mask: bool = False,
        additional_concat_ch: int = 0,  # hdmap
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            concat_padding_mask (`bool`, *optional*, defaults to False):
                Enable concat padding mask
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.concat_padding_mask = concat_padding_mask
        self.additional_concat_ch = additional_concat_ch

        # embeddings
        in_dim = in_dim + 1 if self.concat_padding_mask else in_dim
        self.patch_embedding = nn.Linear(
            in_dim * patch_size[0] * patch_size[1] * patch_size[2], dim
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)
        if additional_concat_ch > 0:
            self.additional_patch_embedding = nn.Linear(
                additional_concat_ch * patch_size[0] * patch_size[1] * patch_size[2],
                dim,
            )

        # blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    ffn_dim,
                    num_heads,
                    cross_attn_norm,
                    eps,
                    i2v=(model_type == "i2v"),
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        self.is_shuffle_op_fused = False

    def initialize_context_parallel(self, cp_group: ProcessGroup | None = None) -> None:
        """
        Set the context parallel group for the network.

        Must be called before preparing cache.
        """
        for block in self.blocks:
            block.set_context_parallel_group(cp_group)

    def initialize_cache(
        self,
        # self attn
        chunk_size: int,
        window_size: int,
        sink_size: int,
        # cross attn
        text_embeddings: Tensor,  # umt5 text embedding. shape [1, 512, 4096]
        img_embeddings: Optional[
            Tensor
        ] = None,  # CLIP image embedding for I2V. shape [1, 256, 1280]
    ) -> WanDiTNetworkCache:
        """
        Initialize the cache for the DiT.
        """
        context_text = self.text_embedding(text_embeddings)
        if self.model_type == "i2v":
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

    def fuse_ops_into_weights(self):
        """
        Fuse some ops into the weights of the network.

        Note this function should be called only after loading the checkpoint.
        """
        self._fuse_shuffle_op_into_head()

    def _fuse_shuffle_op_into_head(self):
        """
        In the WAN model, the patchify operation is
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",

        while the unpatchify operation is
        "b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)"

        This is likely a bug in the WAN model where the last dimension is shuffled after the network.

        To fix this, we could fuse this shuffle op into the last linear layer of the head,
        so that we do not have to do this shuffle op explicitly before returning the result.

        Calling this function to modify the head in place, is equivalent to the following code
        before returning the result:
        ```python
        x = rearrange(
            x,
            "B L (nt nh nw d) -> B L (d nt nh nw)",
            nt=self.patch_size[0],
            nh=self.patch_size[1],
            nw=self.patch_size[2],
            d=self.out_dim,
        ) # [B, L, D]
        ```
        """
        if self.is_shuffle_op_fused:
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
        timesteps: Optional[Tensor],
        block_kv_caches: List[BlockCache],
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        hdmap: Optional[Tensor] = None,
        eager_mode: bool = True,
    ):
        r"""
        Args:
            x (Tensor): Input tensor with shape [B, L, D] after CP. The layout is assumed to be
                "b (t h w) (d nt nh nw)".
            timesteps (Optional[Tensor]): Timesteps with shape [B].
            block_kv_caches (List[BlockCache]): KV caches for the blocks.
            rope_freqs (Tensor): RoPE frequencies with shape [L, 1, 1, D // 2] after CP.
            hdmap (Optional[Tensor]): HDMap condition tensor with shape [B, L, additional_concat_ch] after CP.
                assuming same layout as x.
        """
        assert x.ndim == 3, "x is expected to be 3D tensor with shape [B, L, D]"
        assert rope_freqs.ndim == 4, (
            "rope_freqs is expected to be 4D tensor with shape [L, 1, 1, D // 2]"
        )
        assert timesteps.ndim == 1, (
            "timesteps is expected to be 2D tensor with shape [B]"
        )
        assert self.is_shuffle_op_fused, (
            "needs to call _fuse_shuffle_op_into_head() before running forward"
        )

        # patch embedding
        x = self.patch_embedding(x)  # (B, L, D)

        # patch embedding for hdmap
        if self.additional_concat_ch > 0:
            assert hdmap is not None, (
                "hdmap is expected to be provided for additional concat channels"
            )
            additional_x = self.additional_patch_embedding(hdmap)
            x = x + additional_x  # (B, L, D)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )  # [B, D]
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # [B, 6, D]

        # transformer blocks
        if eager_mode:
            block_kv_caches.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            x = block(
                x=x,
                e=e0,
                rope_freqs=rope_freqs,
                block_kv_cache=block_kv_caches[block_idx],
            )
        if eager_mode:
            block_kv_caches.after_update(current_chunk_idx)

        # head
        x = self.head(x, e.unsqueeze(1))  # (B, L, D)
        return x


def test_basic(i2v: bool = False, use_hdmap: bool = False):
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

    network = WanDiTNetwork(
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
    ).to(device=device, dtype=dtype)
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
        hdmap = torch.randn(
            1, additional_concat_ch, T, H, W, device=device, dtype=dtype
        )
        hdmap = rearrange(
            hdmap,
            "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
            kt=network.patch_size[0],
            kh=network.patch_size[1],
            kw=network.patch_size[2],
        )

    else:
        hdmap = None

    network.initialize_context_parallel()
    network.fuse_ops_into_weights()

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
            hdmap=hdmap,
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
