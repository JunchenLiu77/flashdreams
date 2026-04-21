from dataclasses import dataclass, field
from typing import Literal
from einops import rearrange
from torch import Tensor
import torch

from flashsim.model.video_vae.base import BaseVideoVAE
from flashsim.config import InstantiateConfig


@dataclass
class PixelShuffleVAECache:
    autoregressive_index: int = -1


@dataclass
class PixelShuffleVAEInterfaceConfig(InstantiateConfig["PixelShuffleVAEInterface"]):
    _target: type["PixelShuffleVAEInterface"] = field(
        default_factory=lambda: PixelShuffleVAEInterface
    )

    frame_selection_mode: Literal["first_frame", "last_frame"] = "last_frame"


class PixelShuffleVAEInterface(
    BaseVideoVAE[PixelShuffleVAECache, PixelShuffleVAECache]
):
    def __init__(
        self,
        config: PixelShuffleVAEInterfaceConfig,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config

    def initialize_encode_cache(self) -> PixelShuffleVAECache:
        return PixelShuffleVAECache()

    def encode(self, x: Tensor, cache: PixelShuffleVAECache | None = None) -> Tensor:
        """
        x is expected to be in the format of [..., T, C, H, W], values in range [-1, 1]

        return: [..., T, C, H, W]
        """
        if cache is None:
            cache = self.initialize_encode_cache()
            cache.autoregressive_index = 0
        assert cache.autoregressive_index >= 0, (
            "Autoregressive index must be updated before encoding"
        )

        T = x.shape[-4]

        if self.config.frame_selection_mode == "first_frame":
            if cache.autoregressive_index == 0:
                indices = [0] + list(range(1, T, 4))
            else:
                indices = list(range(0, T, 4))
        elif self.config.frame_selection_mode == "last_frame":
            if cache.autoregressive_index == 0:
                indices = [0] + list(range(4, T, 4))
            else:
                indices = list(range(3, T, 4))
        else:
            raise ValueError(
                f"Invalid frame selection mode: {self.config.frame_selection_mode}"
            )

        x = x[..., indices, :, :, :]
        z = rearrange(x, "... t c (h h8) (w w8) -> ... t (c h8 w8) h w", h8=8, w8=8)
        return z

    def initialize_decode_cache(self) -> PixelShuffleVAECache:
        raise PixelShuffleVAECache()

    def decode(self, z: Tensor, cache: PixelShuffleVAECache | None = None) -> Tensor:
        raise NotImplementedError("Decoding is not supported for PixelShuffleInterface")

    @property
    def temporal_compression_ratio(self) -> int:
        return 4

    @property
    def spatial_compression_ratio(self) -> int:
        return 8
