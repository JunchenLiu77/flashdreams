import pytest

import tyro
from flashdreams.recipes.taehv import (
    TeahvVAEDecoder,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
    PixelShuffleVAEEncoder,
)
from flashdreams.recipes.wan.autoencoder.vae import WanVAEEncoder, WanVAEEncoderConfig


@pytest.mark.parametrize(
    ("config_cls", "target_cls"),
    [
        (PixelShuffleVAEEncoderConfig, PixelShuffleVAEEncoder),
        (TeahvVAEDecoderConfig, TeahvVAEDecoder),
        (WanVAEEncoderConfig, WanVAEEncoder),
    ],
)
def test_video_vae_config_cli_defaults(config_cls: type, target_cls: type) -> None:
    config = tyro.cli(config_cls, args=[])
    assert isinstance(config, config_cls)
    assert config._target is target_cls


def test_pixelshuffle_cli_accepts_frame_selection_override() -> None:
    config = tyro.cli(
        PixelShuffleVAEEncoderConfig,
        args=["--frame-selection-mode", "first_frame"],
    )
    assert config.frame_selection_mode == "first_frame"
