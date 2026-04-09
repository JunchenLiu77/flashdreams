from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashsim.checkpoint.load import load_checkpoint
from flashsim.configs import InstantiateConfig

from flashsim.model.video_dit.base import BaseVideoDiT, denoise, add_noise
from flashsim.model.video_dit.rope import RotaryPositionEmbedding3D
from flashsim.model.video_dit.flow_match import FlowMatchScheduler
from flashsim.model.video_dit.context_parallel_strategy import (
    HierarchicalCPGroups,
    create_hierarchical_cp_groups,
)
from .network import WanDiTNetwork, WanDiTNetworkCache, WanDiTNetworkConfig


AVAILABLE_WAN2_1_CHECKPOINT_PATHS = {
    "self_forcing": "https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt",
}


@dataclass
class WanDiTCondition:
    """
    Condition for the Wan DiT.
    """

    _is_patchified: bool = False


@dataclass
class WanDiTCache:
    """
    Cache for the Wan DiT.
    """

    len_h: (
        int  # number of tokens along the spatial height dimension after patchification
    )
    len_w: (
        int  # number of tokens along the spatial width dimension after patchification
    )
    num_tokens_per_chunk: int  # number of tokens per chunk after CP
    batch_size: int  # batch size
    num_views: int  # number of views

    network_cache: WanDiTNetworkCache
    rope_adapter: RotaryPositionEmbedding3D

    # For KV cache update in the end.
    x0: Tensor | None = None  # clean latent [B, V, pT, pHW, D]
    condition: WanDiTCondition | None = None

    autoregressive_index: int = -1
    _is_patchified: bool = False


@dataclass
class WanDiTConfig(InstantiateConfig["WanDiT"]):
    _target: type["WanDiT"] = field(default_factory=lambda: WanDiT)

    # Network configurations
    network: WanDiTNetworkConfig = field(default_factory=lambda: WanDiTNetworkConfig())
    dtype: torch.dtype = torch.bfloat16

    # RoPE: Default to 1.0 for no extrapolation.
    h_extrapolation_ratio: float = 1.0
    w_extrapolation_ratio: float = 1.0

    # Difussion schedule
    denoising_timesteps: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True

    # Local attn: Number of tokens along T dimension.
    window_size_t: int = 21
    sink_size_t: int = 0

    # Chunk size: Number of tokens along T dimension. (after patchification)
    len_t: int = 3

    # Checkpoint path
    checkpoint_path: str | None = None

    # Noise level for KV cache update.
    context_noise: int = 0

    # Speedup.
    compile_network: bool = True


class WanDiT(BaseVideoDiT[WanDiTCache]):
    """
    Wan DiT for video generation.
    """

    def __init__(
        self, config: WanDiTConfig, device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        # multi-GPU setup
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            self.cp_groups = create_hierarchical_cp_groups(
                world_size=world_size,
                rank=rank,
                V=1,
                T=config.len_t,
                single_group_as_none=True,
            )
        else:
            self.cp_groups = HierarchicalCPGroups(rank=0)

        self.config = config
        self.dtype = config.dtype
        self.device = device

        self.network = WanDiTNetwork(config=self.config.network)
        self.network = self.network.to(device=self.device, dtype=self.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(
            cp_group=self.cp_groups.THW_group,
        )

        if self.config.checkpoint_path is not None:
            _state_dict = load_checkpoint(self.config.checkpoint_path)
            # self-forcing checkpoint
            if "generator_ema" in _state_dict:
                _state_dict = _state_dict["generator_ema"]
            state_dict = {}
            for k, v in _state_dict.items():
                if k.startswith("model."):
                    new_k = k[len("model.") :]
                elif k.startswith("net."):
                    new_k = k[len("net.") :]
                else:
                    new_k = k
                state_dict[new_k] = v
            self.network.load_state_dict(state_dict)
        self.network.update_parameters_after_loading_checkpoint()

        if self.config.compile_network:
            self.network = torch.compile(
                self.network, mode="max-autotune-no-cudagraphs"
            )

        # define scheduler
        num_train_timestep = 1000
        self.scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(num_train_timestep, training=True)
        if self.config.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[
                num_train_timestep
                - torch.tensor(self.config.denoising_timesteps, dtype=torch.long)
            ]
        else:
            self.denoising_step_list = torch.tensor(
                self.config.denoising_timesteps, dtype=torch.long
            )
        self.denoising_step_list = self.denoising_step_list.to(self.device, self.dtype)

    def initialize_cache(
        self,
        height: int,
        width: int,
        text_embeddings: Tensor,  # [B, V, L, D]
        image_embeddings: Tensor | None = None,  # [B, V, L, D]
        view_names: list[str] | None = None,
    ) -> WanDiTNetworkCache:
        """
        Initialize the cache for the video DiT.

        Args:
            height: The video height after VAE spatial compression.
            width: The video width after VAE spatial compression.
            text_embeddings: Text embeddings [B, V, L, D]
            image_embeddings: CLIP Image embeddings [B, V, L, D] or None for text-to-video
            view_names: List of view names.

        Returns:
            The cache for the video DiT.
        """
        # compute size of the tokens after patchification
        len_t = self.config.len_t
        len_h = height // self.config.network.patch_size[1]
        len_w = width // self.config.network.patch_size[2]

        head_dim = self.config.network.dim // self.config.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=len_t,
            len_h=len_h,
            len_w=len_w,
            head_dim=head_dim,
            h_extrapolation_ratio=self.config.h_extrapolation_ratio,
            w_extrapolation_ratio=self.config.w_extrapolation_ratio,
            interleaved=True,
            device=self.device,
        )
        # RoPE CP splits along same dimension as self-attention CP.
        rope_adapter.set_context_parallel_group(cp_group=self.cp_groups.THW_group)

        num_tokens_per_frame = len_h * len_w
        num_tokens_per_chunk = num_tokens_per_frame * len_t
        num_tokens_window_size = num_tokens_per_frame * self.config.window_size_t
        num_tokens_sink_size = num_tokens_per_frame * self.config.sink_size_t
        if self.cp_groups.THW_group is not None:
            num_tokens_per_chunk //= self.cp_groups.THW_group.size()
            num_tokens_window_size //= self.cp_groups.THW_group.size()
            num_tokens_sink_size //= self.cp_groups.THW_group.size()
        network_cache = self.network.initialize_cache(
            chunk_size=num_tokens_per_chunk,
            window_size=num_tokens_window_size,
            sink_size=num_tokens_sink_size,
            text_embeddings=text_embeddings,
            img_embeddings=image_embeddings,
        )

        cache = WanDiTCache(
            len_h=len_h,
            len_w=len_w,
            network_cache=network_cache,
            rope_adapter=rope_adapter,
            num_tokens_per_chunk=num_tokens_per_chunk,
            batch_size=text_embeddings.shape[0],
            num_views=text_embeddings.shape[1],
        )
        cache = self._patchify(cache)
        return cache

    def generate(
        self,
        condition: WanDiTCondition,
        cache: WanDiTCache,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        condition = self._patchify(condition)
        x0 = None  # clean latent
        for denoising_step in self.config.denoising_timesteps:
            timestep = torch.tensor(
                [denoising_step], device=self.device, dtype=self.dtype
            )
            x0 = self._predict_x0(x0, timestep, condition, cache, rng=rng)

        # Postpone KV cache update to the finalization step.
        cache.x0 = x0
        cache.condition = condition

        x0 = self._unpatchify(cache.len_h, cache.len_w, x0)
        return x0

    def finalize(self, cache: WanDiTCache, rng: torch.Generator | None = None) -> None:
        # update kv cache
        timestep = torch.tensor(
            [self.config.context_noise], device=self.device, dtype=self.dtype
        )
        _ = self._predict_x0(cache.x0, timestep, cache.condition, cache, rng=rng)

    def _predict_x0(
        self,
        x0: Tensor | None,  # clean latent [B, V, pT, pHW, D]
        timestep: Tensor,  # [1] or [B]
        condition: WanDiTCondition,
        cache: WanDiTNetworkCache,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, "Index must be updated before predicting flow"
        alpha = self.scheduler.timestep_to_sigma(timestep)

        rope_freqs = cache.rope_adapter.shift_t(
            offset=autoregressive_index * self.config.len_t
        )
        batch_size = cache.batch_size
        num_views = cache.num_views
        len_thw = cache.num_tokens_per_chunk

        token_dim = (
            self.config.network.in_dim
            * self.config.network.patch_size[0]
            * self.config.network.patch_size[1]
            * self.config.network.patch_size[2]
        )
        input_shape = (batch_size, num_views, len_thw, token_dim)

        if x0 is None:
            # pure noise
            noisy_input = torch.randn(
                input_shape, device=self.device, dtype=self.dtype, generator=rng
            )
        else:
            noisy_input = add_noise(x0, alpha, rng=rng)

        # mock predicted flow
        assert noisy_input.shape == input_shape
        predicted_flow = self.network(
            x=noisy_input,
            timesteps=timestep,
            rope_freqs=rope_freqs,
            cache=cache.network_cache,
            current_chunk_idx=autoregressive_index,
            eager_mode=True,
        )

        x0 = denoise(noisy_input, alpha, predicted_flow)

        return x0

    def _patchify(self, x: Tensor | WanDiTCondition | WanDiTCache) -> Tensor:
        process_groups = [
            self.cp_groups.THW_group,
        ]
        cp_dims = [-2]

        if isinstance(x, WanDiTCache):
            if x._is_patchified:
                return x
            else:
                # nothing to do
                x._is_patchified = True
                return x
        if isinstance(x, WanDiTCondition):
            if x._is_patchified:
                return x
            else:
                # nothing to do
                x._is_patchified = True
                return x
        elif isinstance(x, Tensor):
            return self.network.patchify_and_maybe_split_cp(
                x,
                process_groups=process_groups,
                cp_dims=cp_dims,
            )
        else:
            raise ValueError(f"Invalid input type: {type(x)}")

    def _unpatchify(self, len_h: int, len_w: int, x: Tensor) -> Tensor:
        process_groups = [
            self.cp_groups.THW_group,
        ]
        cp_dims = [-2]

        return self.network.unpatchify_and_maybe_gather_cp(
            pH=len_h,
            pW=len_w,
            x=x,
            process_groups=process_groups,
            cp_dims=cp_dims,
        )


# python -m flashsim.model.video_dit.wan2_1.model
if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = WanDiTConfig(
        checkpoint_path=AVAILABLE_WAN2_1_CHECKPOINT_PATHS["self_forcing"],
        network=WanDiTNetworkConfig(
            patch_embedding_type="conv3d",
        ),
    ).setup(device=device)

    text_embeddings = torch.randn(1, 1, 512, 4096, device=device, dtype=dtype)
    cache = model.initialize_cache(
        height=720 // 8,
        width=1280 // 8,
        text_embeddings=text_embeddings,
    )

    with torch.no_grad():
        cache.autoregressive_index = 0
        video = model.generate(condition=WanDiTCondition(), cache=cache)
    print(video.shape)
