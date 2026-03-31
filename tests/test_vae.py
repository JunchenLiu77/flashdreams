from typing import Literal

import mediapy as media
import torch

from flashsim.model.video_vae.teahv import TeahvInterface
from flashsim.model.video_vae.wan import WanVAEInterface


@torch.no_grad()
def test_tokenizer(
    tokenizer_choice: Literal["lightvae", "vae"] = "vae",
    detokenizer_choice: Literal["lighttae", "lightvae", "vae"] = "vae",
) -> None:
    dtype = torch.bfloat16
    device = torch.device("cuda")

    checkpoint_paths = {
        "lighttae": "../imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/Autoencoders/lighttaew2_1.safetensors",
        "lightvae": "../imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/Autoencoders/lightvaew2_1.safetensors",
        "vae": "../imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/Autoencoders/Wan2.1_VAE.safetensors",
    }

    if tokenizer_choice == "lightvae":
        tokenizer = WanVAEInterface(
            checkpoint_paths["lightvae"], use_lightvae=True, dtype=dtype, device=device
        )
    elif tokenizer_choice == "vae":
        tokenizer = WanVAEInterface(
            checkpoint_paths["vae"], use_lightvae=False, dtype=dtype, device=device
        )
    else:
        raise ValueError(f"Invalid tokenizer: {tokenizer}")

    if detokenizer_choice == "lighttae":
        detokenizer = TeahvInterface(
            checkpoint_paths["lighttae"], dtype=dtype, device=device
        )
    elif detokenizer_choice == "lightvae":
        detokenizer = WanVAEInterface(
            checkpoint_paths["lightvae"], use_lightvae=True, dtype=dtype, device=device
        )
    elif detokenizer_choice == "vae":
        detokenizer = WanVAEInterface(
            checkpoint_paths["vae"], use_lightvae=False, dtype=dtype, device=device
        )
    else:
        raise ValueError(f"Invalid detokenizer: {detokenizer}")

    tokenizer_cache = tokenizer.initialize_encode_cache()
    detokenizer_cache = detokenizer.initialize_decode_cache()

    video_path = "../imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/mv_minimal/dataset/replay/comparison_steering_wheel_704p.mp4"
    video = media.read_video(video_path)[:81]  # [T, H, W, 3]
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
