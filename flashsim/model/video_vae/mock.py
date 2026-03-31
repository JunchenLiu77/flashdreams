from dataclasses import dataclass

import torch
from torch import Tensor
from flashsim.model.video_vae.base import BaseVideoVAE


@dataclass
class MockVideoVAEEncoderCache:
    """
    A mock cache for the video VAE encoder.
    """

    autoregressive_index: int = -1


@dataclass
class MockVideoVAEDecoderCache:
    """
    A mock cache for the video VAE decoder.
    """

    autoregressive_index: int = -1


@dataclass
class MockVideoVAEConfig:
    in_channels: int = 3
    hidden_channels: int = 16
    out_channels: int = 3


class MockVideoVAE(BaseVideoVAE[MockVideoVAEEncoderCache, MockVideoVAEDecoderCache]):
    def __init__(
        self,
        config: MockVideoVAEConfig,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

    def initialize_encode_cache(self) -> MockVideoVAEEncoderCache:
        return MockVideoVAEEncoderCache()

    def encode(
        self, x: Tensor, cache: MockVideoVAEEncoderCache | None = None
    ) -> Tensor:
        if cache is None:
            # create a temporary cache
            cache = self.initialize_encode_cache()
            cache.autoregressive_index = 0

        assert x.ndim >= 4, "Expected input tensor to have shape [..., C, T, H, W]"
        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, "Index must be updated before encoding"

        if autoregressive_index == 0:
            # VAE processes [1, 4, 4, 4, ...] frames. For the first chunk, we
            # pad 3 frames to the left.
            frame = x[..., :, :1, :, :]
            x = torch.cat([frame, frame, frame, x], dim=-3)

        *batch_shape, C, T, H, W = x.shape
        assert T % self.temporal_compression_ratio == 0, x.shape
        assert H % self.spatial_compression_ratio == 0
        assert W % self.spatial_compression_ratio == 0
        assert C == self.config.in_channels

        Tl = T // self.temporal_compression_ratio
        Hl = H // self.spatial_compression_ratio
        Wl = W // self.spatial_compression_ratio
        Cl = self.config.hidden_channels

        z = torch.randn(*batch_shape, Cl, Tl, Hl, Wl, device=x.device, dtype=x.dtype)
        return z

    def initialize_decode_cache(self) -> MockVideoVAEDecoderCache:
        return MockVideoVAEDecoderCache()

    def decode(
        self, z: Tensor, cache: MockVideoVAEDecoderCache | None = None
    ) -> Tensor:
        if cache is None:
            # create a temporary cache
            cache = self.initialize_decode_cache()
            cache.autoregressive_index = 0

        assert z.ndim >= 4, "Expected input tensor to have shape [..., Cl, Tl, Hl, Wl]"
        autoregressive_index = cache.autoregressive_index
        assert autoregressive_index >= 0, "Index must be updated before decoding"

        *batch_shape, Cl, Tl, Hl, Wl = z.shape
        assert Cl == self.config.hidden_channels

        T = Tl * self.temporal_compression_ratio
        H = Hl * self.spatial_compression_ratio
        W = Wl * self.spatial_compression_ratio
        C = self.config.out_channels

        x = torch.randn(*batch_shape, C, T, H, W, device=z.device, dtype=z.dtype)

        if autoregressive_index == 0:
            # VAE processes [1, 4, 4, 4, ...] frames. For the first chunk, we
            # crop out the first 3 frames.
            x = x[..., :, 3:, :, :]

        return x

    @property
    def temporal_compression_ratio(self) -> int:
        return 4

    @property
    def spatial_compression_ratio(self) -> int:
        return 8
