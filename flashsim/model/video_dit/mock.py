from dataclasses import dataclass

import torch
from torch import Tensor
from einops import rearrange


from flashsim.model.video_dit.base import BaseVideoDiT, denoise, add_noise


class MockRoPEAdapter:
    def __init__(
        self,
        len_t: int,
        len_h: int,
        len_w: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        self.len_t = len_t
        self.len_h = len_h
        self.len_w = len_w
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

    def get_freqs(self, shift_t: int = 0) -> Tensor:
        L = self.len_t * self.len_h * self.len_w
        return (
            torch.randn(
                L, 1, 1, self.head_dim // 2, device=self.device, dtype=self.dtype
            )
            + shift_t
        )


@dataclass
class MockVideoDiTCondition:
    """
    A mock condition for the video DiT.
    """

    text: Tensor  # text embeddings [B, L, D]
    image: Tensor  # first frame of the video [B, 1, pH, pW, D]
    hdmap: Tensor | None = None  # hdmap of the video [B, pT, pH, pW, D]


@dataclass
class MockVideoDiTCache:
    """
    A mock cache for the video DiT.
    """

    len_h: (
        int  # number of tokens along the spatial height dimension after patchification
    )
    len_w: (
        int  # number of tokens along the spatial width dimension after patchification
    )

    rope_adapter: MockRoPEAdapter
    autoregressive_index: int = -1


@dataclass
class MockVideoDiTConfig:
    head_dim: int = 128
    in_channels: int = 16
    len_t: int = 4  # number of tokens along the temporal dimension after patchification


class MockVideoDiT(BaseVideoDiT[MockVideoDiTCache]):
    """
    A mock video DiT for testing purposes.
    """

    def __init__(
        self,
        config: MockVideoDiTConfig,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

    def initialize_cache(self, height: int, width: int) -> MockVideoDiTCache:
        """
        Initialize the cache for the video DiT.

        Args:
            height: The video height after VAE spatial compression.
            width: The video width after VAE spatial compression.

        Returns:
            The cache for the video DiT.
        """
        # compute size of the tokens after patchification
        len_h = height // self.spatial_patch_size
        len_w = width // self.spatial_patch_size

        rope_adapter = MockRoPEAdapter(
            len_t=self.config.len_t,
            len_h=len_h,
            len_w=len_w,
            head_dim=self.config.head_dim,
        )
        return MockVideoDiTCache(len_h=len_h, len_w=len_w, rope_adapter=rope_adapter)

    def timestep_to_sigma(self, timestep: Tensor) -> Tensor:
        return timestep

    def _predict_x0(
        self,
        x0: Tensor | None,  # clean latent [B, T, H, W, D]
        timestep: Tensor,  # [1] or [B]
        condition: MockVideoDiTCondition,
        cache: MockVideoDiTCache,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        alpha = self.timestep_to_sigma(timestep)

        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, "Index must be updated before predicting flow"

        len_t = self.config.len_t
        len_h = cache.len_h
        len_w = cache.len_w

        num_tokens_per_chunk = len_t * len_h * len_w
        _ = cache.rope_adapter.get_freqs(
            shift_t=autoregressive_index * num_tokens_per_chunk
        )

        batch_size = timestep.shape[0]
        token_dim = (
            self.config.in_channels
            * self.temporal_patch_size
            * self.spatial_patch_size**2
        )
        input_shape = (batch_size, len_t, len_h, len_w, token_dim)

        if x0 is None:
            # pure noise
            noisy_input = torch.randn(
                input_shape, device=self.device, dtype=self.dtype, generator=rng
            )
        else:
            noisy_input = add_noise(x0, alpha, rng=rng)

        # mock predicted flow
        assert noisy_input.shape == input_shape
        predicted_flow = torch.randn_like(noisy_input)

        x0 = denoise(noisy_input, alpha, predicted_flow)
        return x0

    def generate(
        self,
        condition: MockVideoDiTCondition,
        cache: MockVideoDiTCache,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        x0 = None  # clean latent
        for denoising_step in self.denoising_timesteps:
            timestep = torch.tensor(
                [denoising_step], device=self.device, dtype=self.dtype
            )
            x0 = self._predict_x0(x0, timestep, condition, cache, rng=rng)
        return x0

    @property
    def temporal_patch_size(self) -> int:
        return 1

    @property
    def spatial_patch_size(self) -> int:
        return 2

    @property
    def denoising_timesteps(self) -> list[int]:
        return [1000, 750, 500, 250]

    def patchify(self, x: Tensor) -> Tensor:
        """
        Patchify the input tensor.

        The patchify pattern is:
            "... c (t kt) (h kh) (w kw) -> ... t h w (c kt kh kw)"

        Args:
            x: The input tensor. [..., C, T, H, W]

        Returns:
            The patched tensor. [..., len_t, len_h, len_w, D]
        """
        x = rearrange(
            x,
            "... c (t kt) (h kh) (w kw) -> ... t h w (c kt kh kw)",
            kt=self.temporal_patch_size,
            kh=self.spatial_patch_size,
            kw=self.spatial_patch_size,
        )
        return x

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Unpatchify the input tensor.

        The unpatchify pattern is:
            "... t h w (c kt kh kw) -> ... c (t kt) (h kh) (w kw)"

        Args:
            x: The input tensor. [..., len_t, len_h, len_w, D]

        Returns:
            The unpatched tensor. [..., C, T, H, W]
        """
        x = rearrange(
            x,
            "... t h w (c kt kh kw) -> ... c (t kt) (h kh) (w kw)",
            kt=self.temporal_patch_size,
            kh=self.spatial_patch_size,
            kw=self.spatial_patch_size,
        )
        return x
