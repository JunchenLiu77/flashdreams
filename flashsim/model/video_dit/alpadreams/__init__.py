from dataclasses import dataclass, field
from typing import Final

import torch
from torch import Tensor

from flashsim.model.video_dit.base import BaseVideoDiT

from .rope import RotaryPositionEmbedding3D
from .network import CosmosDiT as CosmosDiTNetwork
from .network import CosmosDiTCache as CosmosDiTNetworkCache
from .flow_match import FlowMatchScheduler

from flashsim.checkpoint.load import load_checkpoint

DEFAULT_CAMERAS: Final[tuple[str, ...]] = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

DEFAULT_CAMERA_VIEW_MAPPING: Final = dict(zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS))))


@dataclass
class CosmosDiTCondition:
    """
    Condition for the Cosmos DiT.
    """
    hdmap: Tensor # hdmap of the video [B, V, pT, pHW, D]
    condition_video_input_mask: Tensor # condition video input mask [B, V, pT, pHW, 4]

@dataclass
class CosmosDiTCache:
    """
    Cache for the Cosmos DiT.
    """
    len_h: int # number of tokens along the spatial height dimension after patchification
    len_w: int # number of tokens along the spatial width dimension after patchification

    network_cache: CosmosDiTNetworkCache
    rope_adapter: RotaryPositionEmbedding3D

    encoded_image: Tensor # first frame of the video [B, V, 1, pHW, D]
    view_indices: Tensor | None = None # view indices [B, V]

    autoregressive_index: int = -1


@dataclass
class CosmosDiTConfig:
    # Network configurations
    in_out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    enable_hdmap_condition: bool = True
    encode_with_pixel_shuffle: bool = False
    enable_cross_view_attn: bool = False

    # For 720P set to 3.0; for 480P set to 2.0;
    h_extrapolation_ratio: float = 3.0
    w_extrapolation_ratio: float = 3.0

    # Difussion schedule
    denoising_timesteps: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    warp_denoising_step: bool = True

    # Local attn: Number of tokens along T dimension.
    window_size_t: int = 8
    sink_size_t: int = 0

    # Chunk size: Number of tokens along T dimension. (after patchification)
    len_t: int = 4

    # Checkpoint path
    checkpoint_path: str | None = None

class CosmosDiT(BaseVideoDiT[CosmosDiTCache]):
    """
    Cosmos DiT for video generation.
    """
    def __init__(
        self, 
        config: CosmosDiTConfig, 
        dtype: torch.dtype = torch.bfloat16, 
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

        if self.config.enable_hdmap_condition:
            additional_concat_ch = 192 if self.config.encode_with_pixel_shuffle else 16
        else:
            additional_concat_ch = 0

        self.network = CosmosDiTNetwork(
            in_channels=config.in_out_channels,
            out_channels=config.in_out_channels,
            patch_spatial=config.patch_spatial,
            patch_temporal=config.patch_temporal,
            additional_concat_ch=additional_concat_ch,
            enable_cross_view_attn=self.config.enable_cross_view_attn,
        ).to(self.device, self.dtype)

        if self.config.checkpoint_path is not None:
            state_dict = load_checkpoint(self.config.checkpoint_path)
            for k, v in state_dict.items():
                if k.startswith("net."):
                    state_dict[k.replace("net.", "")] = v
            self.network.load_state_dict(state_dict)
        self.network.update_parameters_after_loading_checkpoint()

        # define scheduler
        num_train_timestep = 1000
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_train_timestep, training=True)
        if self.config.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[
                num_train_timestep - torch.tensor(self.config.denoising_timesteps, dtype=torch.long)
            ]
        else:
            self.denoising_step_list = torch.tensor(self.config.denoising_timesteps, dtype=torch.long)
        self.denoising_step_list = self.denoising_step_list.to(self.device, self.dtype)


    def initialize_cache(
        self, 
        height: int, 
        width: int,
        encoded_image: Tensor, # [B, V, 1, pHW, D] after patchify
        text_embeddings: Tensor, # [B, V, L, D]
        view_names: list[str] | None = None,
    ) -> CosmosDiTNetworkCache:
        """
        Initialize the cache for the video DiT.

        Args:
            height: The video height after VAE spatial compression.
            width: The video width after VAE spatial compression.
            image: First frame of the video after VAE spatial compression [B, V, 1, pHW, D] after patchify
            text_embeddings: Text embeddings [B, V, L, D]
            view_names: List of view names.

        Returns:
            The cache for the video DiT.
        """
        # compute size of the tokens after patchification
        len_h = height // self.spatial_patch_size
        len_w = width // self.spatial_patch_size

        head_dim = self.network.model_channels // self.network.num_heads
        rope_adapter = RotaryPositionEmbedding3D(
            len_t=self.config.len_t, 
            len_h=len_h, 
            len_w=len_w, 
            head_dim=head_dim,
            h_extrapolation_ratio=self.config.h_extrapolation_ratio,
            w_extrapolation_ratio=self.config.w_extrapolation_ratio,
            device=self.device
        )

        num_tokens_per_frame = len_h * len_w
        network_cache = self.network.initialize_cache(
            chunk_size=num_tokens_per_frame * self.config.len_t,
            window_size=num_tokens_per_frame * self.config.window_size_t,
            sink_size=num_tokens_per_frame * self.config.sink_size_t,
            context=text_embeddings,
        )

        view_indices: Tensor | None = None
        if self.config.enable_cross_view_attn:
            batch_size = encoded_image.shape[0]
            assert view_names is not None, "View names must be provided if cross-view attention is enabled"
            view_indices = torch.tensor(
                [DEFAULT_CAMERA_VIEW_MAPPING[name] for name in view_names],
                device=self.device,
                dtype=torch.long,
            )
            view_indices = view_indices.repeat(batch_size, 1)

        return CosmosDiTCache(
            len_h=len_h,
            len_w=len_w,
            view_indices=view_indices,
            encoded_image=encoded_image,
            network_cache=network_cache,
            rope_adapter=rope_adapter
        )

    def timestep_to_sigma(self, timestep: Tensor) -> Tensor:
        return self.scheduler.timestep_to_sigma(timestep)

    def predict_x0(
        self, 
        x0: Tensor | None, # clean latent [B, V, T, HW, D]
        timestep: Tensor, # [1] or [B]
        condition: CosmosDiTCondition, 
        cache: CosmosDiTNetworkCache,
        rng: torch.Generator | None = None
    ) -> Tensor:
        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, "Index must be updated before predicting flow"

        len_t = self.config.len_t
        len_h = cache.len_h
        len_w = cache.len_w

        num_tokens_per_chunk = len_t * len_h * len_w
        rope_freqs = cache.rope_adapter.shift_t(offset=autoregressive_index * num_tokens_per_chunk)

        batch_size = condition.hdmap.shape[0]
        num_views = condition.hdmap.shape[1]
        token_dim = (
            self.config.in_out_channels * self.temporal_patch_size * self.spatial_patch_size ** 2
        )
        input_shape = (batch_size, num_views, len_t, len_h*len_w, token_dim)

        if x0 is None:
            # pure noise
            noisy_input = torch.randn(
                input_shape, device=self.device, dtype=self.dtype, generator=rng
            )
        else:
            noisy_input = self.add_noise(x0, timestep, rng=rng)

        # for first chunk, inject back the conditional image latent.
        mask: Tensor | None = None
        image_latent: Tensor | None = None
        if autoregressive_index == 0:
            mask = condition.condition_video_input_mask[..., :1]
            image_latent = cache.encoded_image
            noisy_input.mul_(1.0 - mask).add_(image_latent * mask)

        # mock predicted flow
        assert noisy_input.shape == input_shape
        predicted_flow = self.network(
            x=noisy_input,
            timesteps=timestep,
            rope_freqs=rope_freqs,
            cache=cache.network_cache,
            condition_video_input_mask=condition.condition_video_input_mask,
            current_chunk_idx=autoregressive_index,
            hdmap_condition=condition.hdmap,
            view_indices=cache.view_indices,
            eager_mode=True,
        )

        x0 = self.denoise(noisy_input, timestep, predicted_flow)

        # for first chunk, inject back the conditional image latent.
        if autoregressive_index == 0:
            x0.mul_(1.0 - mask).add_(image_latent * mask)

        return x0

    def patchify(self, x: Tensor) -> Tensor:
        return self.network.patchify_and_maybe_split_cp(x)

    def unpatchify(self, x: Tensor) -> Tensor:
        return self.network.unpatchify_and_maybe_gather_cp(x)

    @property
    def temporal_patch_size(self) -> int:
        return self.config.patch_temporal

    @property
    def spatial_patch_size(self) -> int:
        return self.config.patch_spatial

    @property
    def denoising_timesteps(self) -> list[int]:
        return self.config.denoising_timesteps