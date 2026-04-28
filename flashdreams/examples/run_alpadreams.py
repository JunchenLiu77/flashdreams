"""Alpadreams demo on top of :class:`AlpadreamsPipeline`.

Distributed streaming inference entrypoint for the alpadreams
driving-scene video generation recipe (Cosmos DiT + HDMap + I2V mask).
Picks one of :data:`ALPADREAMS_CONFIG_BUILDERS` based on
``--n_cameras``:

- ``--n_cameras 1`` — single front-facing camera, defaults to
  ``sv_2steps_chunk2_loc6_lightvae_lighttae``.
- ``--n_cameras 4`` — four surrounding cameras, defaults to
  ``mv_2steps_chunk4_loc8_pshuffle_lighttae``.

Each AR step consumes a per-chunk HDMap pixel tensor (pre-extracted
from the example MP4s) and, at step 0 only, the first-frame pixel
tensor that seeds the I2V mask injection inside
:class:`CosmosTransformer`.

Run::

    # Single front-facing camera
    torchrun --nproc_per_node=N \\
        examples/run_alpadreams.py \\
        --n_cameras 1 \\
        --total_blocks 60

    # 4 surrounding cameras
    torchrun --nproc_per_node=N \\
        examples/run_alpadreams.py \\
        --n_cameras 4 \\
        --total_blocks 60
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch
from einops import rearrange

from flashdreams.core.distributed import init as distributed_init
from flashdreams.core.io.s3_sync import sync_s3_dir_to_local
from flashdreams.recipes.alpadreams.config import (
    ALPADREAMS_CONFIG_BUILDERS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATA_DIR_S3 = "s3://flashdreams/assets/example_data/alpadreams"
EXAMPLE_DATA_DIR_LOCAL = str(REPO_ROOT / "assets/example_data/alpadreams")


def _build_data(n_cameras: int) -> tuple[list[str], list[dict]]:
    """Hardcoded example data per camera count, matching legacy alpadreams."""
    if n_cameras == 1:
        camera_names = ["camera_front_wide_120fov"]
        prompt = (
            "Driving scene from a front-facing car camera. Urban environment with roads, "
            "vehicles, pedestrians, traffic signs, and buildings. Clear visibility, "
            "realistic lighting, photorealistic quality. High resolution dashcam footage "
            "of city driving."
        )
        config_name = "sv_2steps_chunk2_loc6_lightvae_lighttae"
    elif n_cameras == 4:
        camera_names = [
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
            "camera_front_wide_120fov",
        ]
        prompt = (
            "Wide-angle urban street scene from a low, dashboard-level viewpoint. "
            "A straight two-lane road with a faded center line and curbside parking on "
            "both sides. Parked sedans and SUVs in neutral colors line the curbs. On the "
            "right, a white stucco mid-rise building with blue fabric awnings, rectangular "
            "windows, and small storefronts at street level. On the left, a low commercial "
            "strip with dark trim, glass fronts, signage, and shaded sidewalks. Mature green "
            "trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday "
            "sunlight, natural colors, realistic materials, crisp shadows, clean asphalt texture."
        )
        config_name = "mv_2steps_chunk4_loc8_pshuffle_lighttae"
    else:
        raise ValueError(f"Number of cameras must be 1 or 4, got {n_cameras}")

    data = [
        {
            "hdmap_video_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.mp4"),
            "first_frame_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.png"),
            "prompt": prompt,
        }
        for name in camera_names
    ]
    return [config_name, *camera_names], data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_cameras", type=int, default=1, help="Number of cameras (1 or 4)."
    )
    parser.add_argument(
        "--total_blocks", type=int, default=60, help="Total blocks to generate."
    )
    parser.add_argument(
        "--overwrite_config_name",
        type=str,
        default=None,
        choices=sorted(ALPADREAMS_CONFIG_BUILDERS.keys()) + [None],  # type: ignore[arg-type]
        help="Optionally override the per-n_cameras default config name.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile of the DiT network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.n_cameras in (1, 4), "Only 1 or 4 cameras are supported"

    config_meta, data = _build_data(args.n_cameras)
    config_name = (
        args.overwrite_config_name
        if args.overwrite_config_name is not None
        else config_meta[0]
    )
    camera_names = config_meta[1:]

    print(
        f"Running alpadreams inference with {args.n_cameras} cameras and config: "
        f"{config_name}"
    )

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    distributed_init()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print(
        f"initialized distributed inference with world size {world_size} and rank {rank}"
    )
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    credential_path = str(REPO_ROOT / "credentials/s3_checkpoint.secret")
    assert os.path.exists(credential_path), (
        f"Credential file not found at {credential_path}"
    )
    sync_s3_dir_to_local(
        s3_dir=EXAMPLE_DATA_DIR_S3,
        s3_credential_path=credential_path,
        cache_dir=EXAMPLE_DATA_DIR_LOCAL,
        max_workers=10,
        show_progress=True,
        verify_checksum=True,
        desc="Syncing from S3",
    )

    # ``huggingface_hub`` already uses ``HF_TOKEN`` automatically as the
    # bearer for every API call when the env var is set, so calling
    # ``login()`` here is redundant -- and harmful in distributed runs:
    # every rank would hit ``GET /api/whoami-v2`` concurrently from the
    # same IP and trip the per-IP 429 bucket. Just assert the token is
    # present and let the lib pick it up on demand.
    assert os.getenv("HF_TOKEN") is not None, "HF_TOKEN is not set"
    if rank == 0:
        print("HF_TOKEN detected; using env-var auth for huggingface_hub")

    builder = ALPADREAMS_CONFIG_BUILDERS[config_name]
    pipeline_config = builder(
        cp_size=world_size,
        compile_network=not args.no_compile,
        seed=42 + rank,
    )
    pipeline = pipeline_config.setup()
    pipeline.to(device=device)

    # The transformer config bakes in a fixed (latent) resolution. Resize
    # all pixel-space inputs (first frame, HDMap video) to the matching
    # pixel-space resolution before feeding them to the pipeline.
    transformer_cfg = pipeline.diffusion_model.transformer.config
    decoder_sp = pipeline.decoder.SPATIAL_COMPRESSION_RATIO
    pixel_h = transformer_cfg.height * decoder_sp
    pixel_w = transformer_cfg.width * decoder_sp

    first_frames: list[torch.Tensor] = []
    hdmap_videos: list[torch.Tensor] = []
    prompts: list[str] = []
    for entry in data:
        first_frame = media.read_image(entry["first_frame_path"])
        first_frame = cv2.resize(first_frame, (pixel_w, pixel_h))
        first_frame_t = (
            torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
        )
        first_frames.append(rearrange(first_frame_t, "h w c -> 1 c h w"))

        hdmap_video_np = media.read_video(entry["hdmap_video_path"])
        if hdmap_video_np.shape[1:3] != (pixel_h, pixel_w):
            hdmap_video_np = np.stack(
                [cv2.resize(f, (pixel_w, pixel_h)) for f in hdmap_video_np], axis=0
            )
        hdmap_video_t = (
            torch.from_numpy(hdmap_video_np).to(dtype=dtype, device=device) / 127.5
            - 1.0
        )
        hdmap_videos.append(rearrange(hdmap_video_t, "t h w c -> t c h w"))

        prompts.append(entry["prompt"])

    first_frames_t = torch.stack(first_frames, dim=0).unsqueeze(
        0
    )  # [B=1, V, 1, C, H, W]
    hdmap_videos_t = torch.stack(hdmap_videos, dim=0).unsqueeze(
        0
    )  # [B=1, V, T, C, H, W]
    prompts_2d: list[list[str]] = [prompts]  # [B=1, V]
    hdmap_num_frames = hdmap_videos_t.shape[2]
    print("loaded hdmap_videos.shape:", hdmap_videos_t.shape)

    cache = pipeline.initialize_cache(
        text=prompts_2d, image=first_frames_t, view_names=camera_names
    )

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    generated_video: list[torch.Tensor] = []
    start = 0
    for i in range(args.total_blocks):
        num_frames = pipeline.get_num_frames(i)
        end = start + num_frames
        if end > hdmap_num_frames:
            break
        print(
            f"autoregressive_index: {i}, num_frames: {num_frames}, start: {start}, end: {end}"
        )
        generated_video.append(
            pipeline.generate(
                autoregressive_index=i,
                cache=cache,
                hdmap=hdmap_videos_t[:, :, start:end],
            )
        )
        start = end
        pipeline.finalize(i, cache)

    video = torch.cat(generated_video, dim=2)  # [B, V, T, C, H, W]
    generated_num_frames = video.shape[2]
    print("end of streaming inference, generated_video.shape:", video.shape)

    if rank == 0:
        condition = hdmap_videos_t[:, :, :generated_num_frames]
        canvas = rearrange(
            torch.cat([condition, video], dim=-2),
            "1 v t c h w -> t h (v w) c",
        )
        canvas = (canvas.float().cpu().numpy() + 1.0) / 2.0
        canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
        save_path = f"{REPO_ROOT}/outputs/alpadreams_{config_name}_{world_size}gpus.mp4"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, canvas, fps=30)
        print(f"saved generated video to {save_path}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
