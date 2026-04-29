"""TAEHV video decoder, exposed as a infra :class:`Decoder`.

TAEHV is decode-only in our pipelines (encoding raises ``NotImplementedError``
in the underlying impl), so this module ports just the decode side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.decoder import Decoder, DecoderConfig

from .impl import TAEHV, TAEHVCache

AVAILABLE_TAEHV_CHECKPOINT_PATHS = {
    "lighttae": "s3://flashdreams/assets/checkpoints/autoencoders/lighttaew2_1.pth",
}


@dataclass(kw_only=True)
class TeahvVAEDecoderConfig(DecoderConfig):
    _target: type["TeahvVAEDecoder"] = field(default_factory=lambda: TeahvVAEDecoder)

    checkpoint_path: str = AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"]
    parallel: bool = True
    """``True``: faster + higher memory; ``False``: lower memory."""
    dtype: torch.dtype = torch.bfloat16


class TeahvVAEDecoder(Decoder[TAEHVCache]):
    """TAEHV (Tiny AutoEncoder for Hunyuan Video) decoder.

    Forward input is a latent tensor of shape ``[..., Tl, Cl, Hl, Wl]``;
    output is a video tensor of shape ``[..., T, C, H, W]`` with values in
    ``[-1, 1]``.
    """

    TEMPORAL_COMPRESSION_RATIO = 4
    SPATIAL_COMPRESSION_RATIO = 8

    # Per-channel scaling for the lighttae checkpoint.
    _LIGHTTAE_MEAN: tuple[float, ...] = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    _LIGHTTAE_STD: tuple[float, ...] = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )

    def __init__(self, config: TeahvVAEDecoderConfig) -> None:
        super().__init__(config)
        self.config: TeahvVAEDecoderConfig = config

        self.parallel = config.parallel
        self.need_scaled = "lighttae" in config.checkpoint_path
        self.taehv = TAEHV(checkpoint_path=config.checkpoint_path).to(
            dtype=config.dtype
        )

        if self.need_scaled:
            self.register_buffer(
                "mean",
                torch.tensor(self._LIGHTTAE_MEAN, dtype=config.dtype),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor(self._LIGHTTAE_STD, dtype=config.dtype),
                persistent=False,
            )

    def initialize_autoregressive_cache(self) -> TAEHVCache:
        return self.taehv.prepare_cache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: TAEHVCache | None = None,
    ) -> Tensor:
        if cache is None:
            cache = self.initialize_autoregressive_cache()

        assert input.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"

        *batch_shape, T, C, H, W = input.shape
        batch_size = math.prod(batch_shape)
        z = input.reshape(batch_size, T, C, H, W)

        if self.need_scaled:
            z = z * self.std.view(1, 1, -1, 1, 1)
            z = z + self.mean.view(1, 1, -1, 1, 1)

        x = (
            self.taehv.decode_video(z, parallel=self.parallel, cache=cache)
            .mul_(2)
            .sub_(1)
        )
        return x.reshape(*batch_shape, *x.shape[1:])

    @property
    def temporal_compression_ratio(self) -> int:
        return self.TEMPORAL_COMPRESSION_RATIO

    @property
    def spatial_compression_ratio(self) -> int:
        return self.SPATIAL_COMPRESSION_RATIO


if __name__ == "__main__":
    import tyro

    config = tyro.cli(TeahvVAEDecoderConfig)
    model = config.setup()
    print(model)
