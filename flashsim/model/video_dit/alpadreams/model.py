from dataclasses import dataclass, field
from typing import Final

import torch
from torch import Tensor

from flashsim.model.video_dit.base import BaseVideoDiT, denoise, add_noise
from flashsim.checkpoint.load import load_checkpoint
from flashsim.configs import InstantiateConfig

from .rope import RotaryPositionEmbedding3D
from .network import CosmosDiTNetwork, CosmosDiTNetworkCache, CosmosDiTNetworkConfig
from .flow_match import FlowMatchScheduler

DEFAULT_CAMERAS: Final[tuple[str, ...]] = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

DEFAULT_CAMERA_VIEW_MAPPING: Final = dict(
    zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS)))
)


@dataclass
class CosmosDiTCondition:
    """
    Condition for the Cosmos DiT.
    """

    hdmap: Tensor  # hdmap of the video [B, V, T, C, H, W]

    _is_patchified: bool = False


@dataclass
class CosmosDiTCache:
    """
    Cache for the Cosmos DiT.
    """

    len_h: (
        int  # number of tokens along the spatial height dimension after patchification
    )
    len_w: (
        int  # number of tokens along the spatial width dimension after patchification
    )

    network_cache: CosmosDiTNetworkCache
    rope_adapter: RotaryPositionEmbedding3D

    image: Tensor  # first frame of the video [B, V, 1, C, H, W]
    condition_video_input_mask_first_block: (
        Tensor  # condition video input mask [B, V, T, 1, H, W]
    )
    condition_video_input_mask_other_blocks: (
        Tensor  # condition video input mask [B, V, T, 1, H, W]
    )
    view_indices: Tensor | None = None  # view indices [B, V]

    # For KV cache update in the end.
    x0: Tensor | None = None  # clean latent [B, V, pT, pHW, D]
    condition: CosmosDiTCondition | None = None

    autoregressive_index: int = -1
    _is_patchified: bool = False


@dataclass
class CosmosDiTConfig(InstantiateConfig["CosmosDiT"]):
    _target: type["CosmosDiT"] = field(default_factory=lambda: CosmosDiT)

    # Network configurations
    enable_hdmap_condition: bool = True
    encode_with_pixel_shuffle: bool = False
    enable_cross_view_attn: bool = False
    network: CosmosDiTNetworkConfig = field(
        default_factory=lambda: CosmosDiTNetworkConfig()
    )

    # For 720P set to 3.0; for 480P set to 2.0;
    h_extrapolation_ratio: float = 3.0
    w_extrapolation_ratio: float = 3.0

    # Difussion schedule
    denoising_timesteps: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True

    # Local attn: Number of tokens along T dimension.
    window_size_t: int = 8
    sink_size_t: int = 0

    # Chunk size: Number of tokens along T dimension. (after patchification)
    len_t: int = 4

    # Checkpoint path
    checkpoint_path: str | None = None

    # Noise level for KV cache update.
    context_noise: int = 128

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.enable_hdmap_condition:
            self.network.additional_concat_ch = (
                192 if self.encode_with_pixel_shuffle else 16
            )
        else:
            self.network.additional_concat_ch = 0


class CosmosDiT(BaseVideoDiT[CosmosDiTCache]):
    """
    Cosmos DiT for video generation.
    """

    def __init__(self, config: CosmosDiTConfig):
        super().__init__()
        self.config = config
        self.dtype = config.dtype
        self.device = config.device

        self.network = CosmosDiTNetwork(config=self.config.network).to(
            self.device, self.dtype
        )

        if self.config.checkpoint_path is not None:
            state_dict = load_checkpoint(self.config.checkpoint_path)
            for k, v in state_dict.items():
                if k.startswith("net."):
                    state_dict[k.replace("net.", "")] = v
            self.network.load_state_dict(state_dict)
        self.network.update_parameters_after_loading_checkpoint()

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
        encoded_image: Tensor,  # [B, V, 1, C, H, W] after VAE spatial compression
        text_embeddings: Tensor,  # [B, V, L, D]
        view_names: list[str] | None = None,
    ) -> CosmosDiTNetworkCache:
        """
        Initialize the cache for the video DiT.

        Args:
            height: The video height after VAE spatial compression.
            width: The video width after VAE spatial compression.
            image: First frame of the video after VAE spatial compression [B, V, 1, C, H, W]
            text_embeddings: Text embeddings [B, V, L, D]
            view_names: List of view names.

        Returns:
            The cache for the video DiT.
        """
        # compute size of the tokens after patchification
        len_t = self.config.len_t
        len_h = height // self.config.network.patch_spatial
        len_w = width // self.config.network.patch_spatial

        head_dim = self.config.network.model_channels // self.config.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=len_t,
            len_h=len_h,
            len_w=len_w,
            head_dim=head_dim,
            h_extrapolation_ratio=self.config.h_extrapolation_ratio,
            w_extrapolation_ratio=self.config.w_extrapolation_ratio,
            device=self.device,
        )

        num_tokens_per_frame = len_h * len_w
        network_cache = self.network.initialize_cache(
            chunk_size=num_tokens_per_frame * len_t,
            window_size=num_tokens_per_frame * self.config.window_size_t,
            sink_size=num_tokens_per_frame * self.config.sink_size_t,
            context=text_embeddings,
        )

        view_indices: Tensor | None = None
        if self.config.enable_cross_view_attn:
            batch_size = encoded_image.shape[0]
            assert view_names is not None, (
                "View names must be provided if cross-view attention is enabled"
            )
            view_indices = torch.tensor(
                [DEFAULT_CAMERA_VIEW_MAPPING[name] for name in view_names],
                device=self.device,
                dtype=torch.long,
            )
            view_indices = view_indices.repeat(batch_size, 1)

        B, V, _, _, H, W = encoded_image.shape
        condition_video_input_mask_first_block = torch.zeros(
            B, V, len_t, 1, H, W, device=self.device, dtype=self.dtype
        )
        condition_video_input_mask_first_block[:, :, :1, :, :, :] = 1.0
        condition_video_input_mask_other_blocks = torch.zeros(
            B, V, len_t, 1, H, W, device=self.device, dtype=self.dtype
        )

        cache = CosmosDiTCache(
            len_h=len_h,
            len_w=len_w,
            view_indices=view_indices,
            image=encoded_image,
            network_cache=network_cache,
            rope_adapter=rope_adapter,
            condition_video_input_mask_first_block=condition_video_input_mask_first_block,
            condition_video_input_mask_other_blocks=condition_video_input_mask_other_blocks,
        )
        cache = self._patchify(cache)
        return cache

    def generate(
        self,
        condition: CosmosDiTCondition,
        cache: CosmosDiTCache,
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

    def finalize(
        self, cache: CosmosDiTCache, rng: torch.Generator | None = None
    ) -> None:
        # update kv cache
        timestep = torch.tensor(
            [self.config.context_noise], device=self.device, dtype=self.dtype
        )
        _ = self._predict_x0(cache.x0, timestep, cache.condition, cache, rng=rng)

    def _predict_x0(
        self,
        x0: Tensor | None,  # clean latent [B, V, pT, pHW, D]
        timestep: Tensor,  # [1] or [B]
        condition: CosmosDiTCondition,
        cache: CosmosDiTNetworkCache,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, "Index must be updated before predicting flow"
        alpha = self.scheduler.timestep_to_sigma(timestep)

        len_t = self.config.len_t
        len_h = cache.len_h
        len_w = cache.len_w

        rope_freqs = cache.rope_adapter.shift_t(offset=autoregressive_index * len_t)

        batch_size = condition.hdmap.shape[0]
        num_views = condition.hdmap.shape[1]
        token_dim = (
            self.config.network.in_channels
            * self.config.network.patch_temporal
            * self.config.network.patch_spatial**2
        )
        input_shape = (batch_size, num_views, len_t, len_h * len_w, token_dim)

        if x0 is None:
            # pure noise
            noisy_input = torch.randn(
                input_shape, device=self.device, dtype=self.dtype, generator=rng
            )
        else:
            noisy_input = add_noise(x0, alpha, rng=rng)

        if autoregressive_index == 0:
            condition_video_input_mask = cache.condition_video_input_mask_first_block
        else:
            condition_video_input_mask = cache.condition_video_input_mask_other_blocks

        # for first chunk, inject back the conditional image latent.
        mask: Tensor | None = None
        image_latent: Tensor | None = None
        if autoregressive_index == 0:
            mask = condition_video_input_mask[..., :1]
            image_latent = cache.image
            noisy_input.mul_(1.0 - mask).add_(image_latent * mask)

        # mock predicted flow
        assert noisy_input.shape == input_shape
        predicted_flow = self.network(
            x=noisy_input,
            timesteps=timestep,
            rope_freqs=rope_freqs,
            cache=cache.network_cache,
            condition_video_input_mask=condition_video_input_mask,
            current_chunk_idx=autoregressive_index,
            hdmap_condition=condition.hdmap,
            view_indices=cache.view_indices,
            eager_mode=True,
        )

        x0 = denoise(noisy_input, alpha, predicted_flow)

        # for first chunk, inject back the conditional image latent.
        if autoregressive_index == 0:
            x0.mul_(1.0 - mask).add_(image_latent * mask)

        return x0

    def _patchify(self, x: Tensor | CosmosDiTCondition | CosmosDiTCache) -> Tensor:
        if isinstance(x, CosmosDiTCache):
            if x._is_patchified:
                return x
            else:
                x.image = self.network.patchify_and_maybe_split_cp(x.image)
                x.condition_video_input_mask_first_block = (
                    self.network.patchify_and_maybe_split_cp(
                        x.condition_video_input_mask_first_block
                    )
                )
                x.condition_video_input_mask_other_blocks = (
                    self.network.patchify_and_maybe_split_cp(
                        x.condition_video_input_mask_other_blocks
                    )
                )
                x._is_patchified = True
                return x
        if isinstance(x, CosmosDiTCondition):
            if x._is_patchified:
                return x
            else:
                x.hdmap = self.network.patchify_and_maybe_split_cp(x.hdmap)
                x._is_patchified = True
                return x
        elif isinstance(x, Tensor):
            return self.network.patchify_and_maybe_split_cp(x)
        else:
            raise ValueError(f"Invalid input type: {type(x)}")

    def _unpatchify(self, len_h: int, len_w: int, x: Tensor) -> Tensor:
        return self.network.unpatchify_and_maybe_gather_cp(pH=len_h, pW=len_w, x=x)


# python -m flashsim.model.video_dit.alpadreams.model
if __name__ == "__main__":
    import tyro

    config = tyro.cli(CosmosDiTConfig)
    model = CosmosDiT(config=config)
