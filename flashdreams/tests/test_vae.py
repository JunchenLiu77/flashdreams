from typing import Literal

import mediapy
import pytest
import torch

from flashdreams.recipes.taehv import (
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)


@torch.no_grad()
@pytest.mark.parametrize("tokenizer_choice", ["lightvae", "vae"])
@pytest.mark.parametrize("detokenizer_choice", ["lighttae", "lightvae", "vae"])
def test_tokenizer(
    tokenizer_choice: Literal["lightvae", "vae"],
    detokenizer_choice: Literal["lighttae", "lightvae", "vae"],
) -> None:
    dtype = torch.bfloat16
    device = torch.device("cuda")

    if tokenizer_choice == "lightvae":
        tokenizer = (
            WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
                dtype=dtype,
            )
            .setup()
            .to(device)
        )
    elif tokenizer_choice == "vae":
        tokenizer = (
            WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
                dtype=dtype,
            )
            .setup()
            .to(device)
        )
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer}")

    if detokenizer_choice == "lighttae":
        detokenizer = (
            TeahvVAEDecoderConfig(
                checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
                dtype=dtype,
            )
            .setup()
            .to(device)
        )
    elif detokenizer_choice == "lightvae":
        detokenizer = (
            WanVAEDecoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
                dtype=dtype,
            )
            .setup()
            .to(device)
        )
    elif detokenizer_choice == "vae":
        detokenizer = (
            WanVAEDecoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
                dtype=dtype,
            )
            .setup()
            .to(device)
        )
    else:
        raise ValueError(f"Invalid detokenizer: {detokenizer}")

    tokenizer_cache = tokenizer.initialize_autoregressive_cache()
    detokenizer_cache = detokenizer.initialize_autoregressive_cache()

    video_path = "./assets/example_data/alpadreams/camera_front_wide_120fov.mp4"
    video = mediapy.read_video(video_path)[:81]  # [T, H, W, 3]
    video = (
        torch.from_numpy(video).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # range [-1, 1]

    video = video.permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]
    encoded_video = tokenizer(video, cache=tokenizer_cache)
    decoded_video = detokenizer(encoded_video, cache=detokenizer_cache)

    l1_loss = torch.nn.functional.l1_loss(video, decoded_video)
    print(
        f"tokenizer: {tokenizer_choice}, detokenizer: {detokenizer_choice}, L1 loss: {l1_loss.item()}"
    )


# python tests/test_vae.py
if __name__ == "__main__":
    for tokenizer_choice in ["lightvae", "vae"]:
        for detokenizer_choice in ["lighttae", "lightvae", "vae"]:
            test_tokenizer(tokenizer_choice, detokenizer_choice)
