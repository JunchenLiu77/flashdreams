"""Pixel-shuffle pseudo-VAE encoder for alpadreams, adapted to the infra :class:`Encoder` API.

Encode-only: there is no learned model; it just selects frames according to
the AR step and pixel-unshuffles each frame's spatial blocks into channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from einops import rearrange
from torch import Tensor

from flashdreams.infra.encoder import (
    Encoder,
    EncoderAutoregressiveCache,
    EncoderConfig,
)


@dataclass(kw_only=True)
class PixelShuffleVAEEncoderCache(EncoderAutoregressiveCache):
    """AR cache for :class:`PixelShuffleVAEEncoder`.

    Tracks ``autoregressive_index`` because the temporal frame-selection rule
    depends on whether this is the first AR step.
    """

    autoregressive_index: int = -1


@dataclass(kw_only=True)
class PixelShuffleVAEEncoderConfig(EncoderConfig):
    _target: type["PixelShuffleVAEEncoder"] = field(
        default_factory=lambda: PixelShuffleVAEEncoder
    )

    frame_selection_mode: Literal["first_frame", "last_frame"] = "last_frame"


class PixelShuffleVAEEncoder(Encoder[PixelShuffleVAEEncoderCache]):
    """Pixel-shuffle pseudo-encoder used as a stand-in for a learned VAE.

    Forward input is a video of shape ``[..., T, C, H, W]`` (range ``[-1, 1]``);
    output is the "latent" of shape ``[..., Tl, Cl, Hl, Wl]`` where ``Cl =
    C * 64`` (8x8 spatial unshuffle) and ``Tl`` depends on the frame
    selection mode and the current AR step.
    """

    TEMPORAL_COMPRESSION_RATIO = 4
    SPATIAL_COMPRESSION_RATIO = 8

    def __init__(self, config: PixelShuffleVAEEncoderConfig) -> None:
        super().__init__(config)
        self.config: PixelShuffleVAEEncoderConfig = config

    def initialize_autoregressive_cache(self) -> PixelShuffleVAEEncoderCache:
        return PixelShuffleVAEEncoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: PixelShuffleVAEEncoderCache | None = None,
    ) -> Tensor:
        if cache is None:
            cache = self.initialize_autoregressive_cache()
        cache.autoregressive_index = autoregressive_index

        T = input.shape[-4]

        if self.config.frame_selection_mode == "first_frame":
            if autoregressive_index == 0:
                indices = [0] + list(range(1, T, 4))
            else:
                indices = list(range(0, T, 4))
        elif self.config.frame_selection_mode == "last_frame":
            if autoregressive_index == 0:
                indices = [0] + list(range(4, T, 4))
            else:
                indices = list(range(3, T, 4))
        else:
            raise ValueError(
                f"Invalid frame selection mode: {self.config.frame_selection_mode}"
            )

        x = input[..., indices, :, :, :]
        return rearrange(x, "... t c (h h8) (w w8) -> ... t (c h8 w8) h w", h8=8, w8=8)

    @property
    def temporal_compression_ratio(self) -> int:
        return self.TEMPORAL_COMPRESSION_RATIO

    @property
    def spatial_compression_ratio(self) -> int:
        return self.SPATIAL_COMPRESSION_RATIO


if __name__ == "__main__":
    import tyro

    config = tyro.cli(PixelShuffleVAEEncoderConfig)
    model = config.setup()
    print(model)
