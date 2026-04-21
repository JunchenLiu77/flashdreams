from typing import Literal

import pytest
import torch
import mediapy

from flashsim.model.video_vae.teahv import (
    TeahvInterfaceConfig,
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
)
from flashsim.model.video_vae.wan import (
    WanVAEInterfaceConfig,
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
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
        tokenizer = WanVAEInterfaceConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
            dtype=dtype,
        ).setup(device=device)
    elif tokenizer_choice == "vae":
        tokenizer = WanVAEInterfaceConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            dtype=dtype,
        ).setup(device=device)
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer}")

    if detokenizer_choice == "lighttae":
        detokenizer = TeahvInterfaceConfig(
            checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
            dtype=dtype,
        ).setup(device=device)
    elif detokenizer_choice == "lightvae":
        detokenizer = WanVAEInterfaceConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
            dtype=dtype,
        ).setup(device=device)
    elif detokenizer_choice == "vae":
        detokenizer = WanVAEInterfaceConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            dtype=dtype,
        ).setup(device=device)
    else:
        raise ValueError(f"Invalid detokenizer: {detokenizer}")

    tokenizer_cache = tokenizer.initialize_encode_cache()
    detokenizer_cache = detokenizer.initialize_decode_cache()

    video_path = "./assets/example_data/alpadreams/camera_front_wide_120fov.mp4"
    video = mediapy.read_video(video_path)[:81]  # [T, H, W, 3]
    video = (
        torch.from_numpy(video).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # range [-1, 1]

    video = video.permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]
    encoded_video = tokenizer.encode(video, tokenizer_cache)
    decoded_video = detokenizer.decode(encoded_video, detokenizer_cache)

    l1_loss = torch.nn.functional.l1_loss(video, decoded_video)
    print(
        f"tokenizer: {tokenizer_choice}, detokenizer: {detokenizer_choice}, L1 loss: {l1_loss.item()}"
    )


# python tests/test_vae.py
if __name__ == "__main__":
    for tokenizer_choice in ["lightvae", "vae"]:
        for detokenizer_choice in ["lighttae", "lightvae", "vae"]:
            test_tokenizer(tokenizer_choice, detokenizer_choice)
