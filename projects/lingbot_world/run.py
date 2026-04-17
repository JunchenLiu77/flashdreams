import os
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
import mediapy as media
from einops import rearrange
from huggingface_hub import login as huggingface_login

from flashsim.distributed import init as distributed_init
from flashsim.io.s3_sync import sync_s3_dir_to_local
from flashsim.model.video_dit.profiling import ProfileEvents

from projects.lingbot_world.config import LINGBOT_WORLD_CONFIGS
from projects.lingbot_world.camera_utils import (
    compute_relative_poses,
    get_plucker_embeddings,
    compute_relative_poses_causal,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--total_blocks", type=int, default=60, help="Total blocks to generate."
)
parser.add_argument(
    "--overwrite_config_name", type=str, default=None, help="Overwrite config name."
)
parser.add_argument("--video_height", type=int, default=464, help="Video height.")
parser.add_argument("--video_width", type=int, default=832, help="Video width.")
args = parser.parse_args()

_REPO_ROOT = Path(__file__).resolve().parents[2]

EXAMPLE_DATA_DIR_S3 = "s3://flashsim/assets/example_data/lingbot_world"
EXAMPLE_DATA_DIR_LOCAL = str(_REPO_ROOT / "assets/example_data/lingbot_world")

CAMERA_NAMES = ["default"]
DATA = [
    {
        "pose_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "poses.npy"),
        "intrinsic_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "intrinsics.npy"),
        "first_frame_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "image.jpg"),
        "text_prompt_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "prompt.txt"),
    }
    for _ in CAMERA_NAMES
]
CONFIG_NAME = "LingBot-World-Fast"

if args.overwrite_config_name is not None:
    CONFIG_NAME = args.overwrite_config_name
print(f"Running Lingbot World inference with config: {CONFIG_NAME}")

# initialize distributed inference
local_rank = int(os.getenv("LOCAL_RANK", 0))
distributed_init()
world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()
print(f"initialized distributed training with world size {world_size} and rank {rank}")
device = torch.device(f"cuda:{local_rank}")
dtype = torch.bfloat16

# login huggingface
if rank == 0:
    HF_TOKEN = os.getenv("HF_TOKEN")
    assert HF_TOKEN is not None, "HF_TOKEN is not set"
    huggingface_login(HF_TOKEN)
    print("logged in to huggingface")

    # download example data from S3
    CREDENTIAL_PATH = str(_REPO_ROOT / "credentials/s3_checkpoint.secret")
    assert os.path.exists(CREDENTIAL_PATH), (
        f"Credential file not found at {CREDENTIAL_PATH}"
    )
    sync_s3_dir_to_local(
        s3_dir=EXAMPLE_DATA_DIR_S3,
        s3_credential_path=CREDENTIAL_PATH,
        cache_dir=EXAMPLE_DATA_DIR_LOCAL,
        max_workers=10,
        show_progress=True,
        verify_checksum=True,
        desc="Syncing from S3",
    )

if torch.distributed.is_initialized():
    torch.distributed.barrier()

# prepare data
plucker_videos = []
camera_intrinsics = []
camera_poses = []
first_frames = []
prompts = []
for data in DATA:
    first_frame = media.read_image(data["first_frame_path"])
    first_frame = cv2.resize(first_frame, (args.video_width, args.video_height))
    first_frame = (
        torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # range [-1, 1]
    first_frame = rearrange(first_frame, "h w c -> 1 c h w")  # [1, C, H, W]
    first_frames.append(first_frame)

    Ks = np.load(data["intrinsic_path"])  # [T, 4]
    Ks = torch.from_numpy(Ks).to(device=device, dtype=torch.float32)
    camera_intrinsics.append(Ks)
    c2ws = np.load(data["pose_path"])  # [T, 4, 4]
    c2ws = torch.from_numpy(c2ws).to(device=device, dtype=torch.float32)
    camera_poses.append(c2ws)
    c2ws, trans_normalizer = compute_relative_poses(c2ws, framewise=True)
    plucker_video = get_plucker_embeddings(
        c2ws, Ks, args.video_height, args.video_width
    )
    plucker_video = rearrange(plucker_video, "t h w c -> t c h w")  # [T, C, H, W]
    plucker_videos.append(plucker_video.to(dtype=dtype))

    prompt = open(data["text_prompt_path"], "r").readlines()[0]
    prompts.append(prompt)
first_frames = torch.stack(first_frames, dim=0).unsqueeze(0)  # [B, V, 1, C, H, W]
plucker_videos = torch.stack(plucker_videos, dim=0).unsqueeze(0)  # [B, V, T, C, H, W]
camera_intrinsics = torch.stack(camera_intrinsics, dim=0).unsqueeze(0)  # [B, V, T, 4]
camera_poses = torch.stack(camera_poses, dim=0).unsqueeze(0)  # [B, V, T, 4, 4]
prompts = [prompts]  # [B, V]
batch_size, num_views, plucker_num_frames, _3, height, width = plucker_videos.shape
print("loaded plucker_videos.shape:", plucker_videos.shape)

# initialize pipeline
pipeline_config = LINGBOT_WORLD_CONFIGS[CONFIG_NAME]
pipeline_config.seed += rank
pipeline = pipeline_config.setup(device=device)
cache = pipeline.initialize_cache(text=prompts, image=first_frames)

torch.cuda.synchronize()
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# streaming inference
start = 0
generated_video = []
last_pose = None
for i in range(args.total_blocks):
    num_frames = pipeline.get_num_frames(i)
    end = start + num_frames
    if end > plucker_num_frames:
        break
    print(
        f"autoregressive_index: {i}, num_frames: {num_frames}, start: {start}, end: {end}"
    )

    curr_intrinsics = camera_intrinsics.squeeze(0).squeeze(0)[start:end]
    curr_poses = camera_poses.squeeze(0).squeeze(0)[start:end]
    curr_poses = compute_relative_poses_causal(curr_poses, trans_normalizer, last_pose)
    last_pose = curr_poses[-1:]

    curr_plucker = get_plucker_embeddings(curr_poses, curr_intrinsics, height, width)
    curr_plucker = rearrange(curr_plucker, "t h w c -> t c h w").to(dtype=dtype)
    curr_plucker = curr_plucker.unsqueeze(0).unsqueeze(0)

    # _ref_plucker = plucker_videos[:, :, start:end]
    # torch.testing.assert_close(curr_plucker, _ref_plucker, atol=1e-3, rtol=1e-3)

    generated_video.append(
        pipeline.streaming_inference(
            autoregressive_index=i,
            plucker=curr_plucker,
            cache=cache,
        )
    )
    start = end
    pipeline.finalize(
        autoregressive_index=i,
        cache=cache,
    )  # update KV cache for the next block
generated_video = torch.cat(generated_video, dim=2)  # [B, V, T, C, H, W], range [-1, 1]
generated_num_frames = generated_video.shape[2]
print("end of streaming inference, generated_video.shape:", generated_video.shape)

if rank == 0:
    # print profiling results.
    torch.cuda.synchronize()
    ProfileEvents.finalize(cache.profile_events, skip_first_n=3)

    # export result
    canvas = rearrange(generated_video, "1 v t c h w -> t h (v w) c")
    canvas = (canvas.float().cpu().numpy() + 1.0) / 2.0  # range [0, 1]
    canvas = (canvas * 255).astype(np.uint8)
    save_path = f"outputs/{CONFIG_NAME}_{world_size}gpus.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    media.write_video(save_path, canvas, fps=16)
    print(f"saved generated video to {save_path}")

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
