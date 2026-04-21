import pytest

import tyro
from flashsim.model.video_vae.pshuffle import (
    PixelShuffleVAEInterface,
    PixelShuffleVAEInterfaceConfig,
)
from flashsim.model.video_vae.teahv import TeahvInterface, TeahvInterfaceConfig
from flashsim.model.video_vae.wan import WanVAEInterface, WanVAEInterfaceConfig


@pytest.mark.parametrize(
    ("config_cls", "target_cls"),
    [
        (PixelShuffleVAEInterfaceConfig, PixelShuffleVAEInterface),
        (TeahvInterfaceConfig, TeahvInterface),
        (WanVAEInterfaceConfig, WanVAEInterface),
    ],
)
def test_video_vae_config_cli_defaults(config_cls: type, target_cls: type) -> None:
    config = tyro.cli(config_cls, args=[])
    assert isinstance(config, config_cls)
    assert config._target is target_cls


def test_pixelshuffle_cli_accepts_frame_selection_override() -> None:
    config = tyro.cli(
        PixelShuffleVAEInterfaceConfig,
        args=["--frame-selection-mode", "first_frame"],
    )
    assert config.frame_selection_mode == "first_frame"
