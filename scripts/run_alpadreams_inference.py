import os
import argparse

import torch
import numpy as np
import mediapy as media
from einops import rearrange
from huggingface_hub import login as huggingface_login

from flashsim.distributed import init as distributed_init
from flashsim.configs.alpadreams import ALPADREAMS_CONFIGS
from flashsim.io.s3_sync import sync_s3_dir_to_local

parser = argparse.ArgumentParser()
parser.add_argument("--n_cameras", type=int, default=1, help="Number of cameras.")
parser.add_argument(
    "--total_blocks", type=int, default=60, help="Total blocks to generate."
)
parser.add_argument(
    "--overwrite_config_name", type=str, default=None, help="Overwrite config name."
)
args = parser.parse_args()
assert args.n_cameras in [1, 4], "Only support 1 or 4 cameras"

EXAMPLE_DATA_DIR_S3 = "s3://flashsim/assets/example_data/alpadreams"
EXAMPLE_DATA_DIR_LOCAL = os.path.join(
    os.path.dirname(__file__), "../assets/example_data/alpadreams"
)

if args.n_cameras == 1:
    CAMERA_NAMES = ["camera_front_wide_120fov"]
    DATA = [
        {
            "hdmap_video_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.mp4"),
            "first_frame_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.png"),
            "prompt": (
                "Driving scene from a front-facing car camera. Urban environment with roads, vehicles, pedestrians, "
                "traffic signs, and buildings. Clear visibility, realistic lighting, photorealistic quality. "
                "High resolution dashcam footage of city driving."
            ),
        }
        for name in CAMERA_NAMES
    ]
    CONFIG_NAME = "sv_2steps_chunk2_loc6_lightvae_lighttae"
elif args.n_cameras == 4:
    CAMERA_NAMES = [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
        "camera_front_wide_120fov",
    ]
    DATA = [
        {
            "hdmap_video_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.mp4"),
            "first_frame_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.png"),
            "prompt": (
                "Wide-angle urban street scene from a low, dashboard-level viewpoint. A straight two-lane road with a faded center line and curbside parking on both sides. "
                "Parked sedans and SUVs in neutral colors line the curbs. On the right, a white stucco mid-rise building with blue fabric awnings, rectangular windows, "
                "and small storefronts at street level. On the left, a low commercial strip with dark trim, glass fronts, signage, and shaded sidewalks. "
                "Mature green trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday sunlight, natural colors, realistic materials, crisp shadows, "
                "clean asphalt texture."
            ),
        }
        for name in CAMERA_NAMES
    ]
    CONFIG_NAME = "mv_2steps_chunk4_loc8_pshuffle_lighttae"
else:
    raise ValueError(f"Number of cameras must be 1 or 4, got {args.n_cameras}")

if args.overwrite_config_name is not None:
    CONFIG_NAME = args.overwrite_config_name
print(
    f"Running Alpadreams inference with {args.n_cameras} cameras and config: {CONFIG_NAME}"
)

# download example data from S3
CREDENTIAL_PATH = os.path.join(
    os.path.dirname(__file__), "../credentials/s3_checkpoint.secret"
)
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

# login huggingface
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF_TOKEN is not set"
huggingface_login(HF_TOKEN)
print("logged in to huggingface")

# initialize distributed inference
distributed_init()
world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()
print(f"initialized distributed training with world size {world_size} and rank {rank}")
device = torch.device(f"cuda:{rank}")
dtype = torch.bfloat16

# prepare data
first_frames = []
hdmap_videos = []
prompts = []
for data in DATA:
    first_frame = media.read_image(data["first_frame_path"])
    first_frame = (
        torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # range [-1, 1]
    first_frame = rearrange(first_frame, "h w c -> 1 c h w")  # [1, C, H, W]
    first_frames.append(first_frame)

    hdmap_video = media.read_video(data["hdmap_video_path"])
    hdmap_video = (
        torch.from_numpy(hdmap_video).to(dtype=dtype, device=device) / 127.5 - 1.0
    )  # range [-1, 1]
    hdmap_video = rearrange(hdmap_video, "t h w c -> t c h w")  # [T, C, H, W]
    hdmap_videos.append(hdmap_video)

    prompts.append(data["prompt"])
first_frames = torch.stack(first_frames, dim=0).unsqueeze(0)  # [B, V, 1, C, H, W]
hdmap_videos = torch.stack(hdmap_videos, dim=0).unsqueeze(0)  # [B, V, T, C, H, W]
prompts = [prompts]  # [B, V]
batch_size, num_views, hdmap_num_frames, _3, height, width = hdmap_videos.shape
print("loaded hdmap_videos.shape:", hdmap_videos.shape)

if torch.distributed.is_initialized():
    torch.distributed.barrier()

# initialize pipeline
pipeline = ALPADREAMS_CONFIGS[CONFIG_NAME].setup(device=device)
cache = pipeline.initialize_cache(
    text=prompts, image=first_frames, view_names=CAMERA_NAMES
)

# streaming inference
start = 0
generated_video = []
for i in range(args.total_blocks):
    num_frames = pipeline.get_num_frames(i)
    end = start + num_frames
    if end > hdmap_num_frames:
        break
    print(
        f"autoregressive_index: {i}, num_frames: {num_frames}, start: {start}, end: {end}"
    )
    generated_video.append(
        pipeline.streaming_inference(
            autoregressive_index=i, hdmap=hdmap_videos[:, :, start:end], cache=cache
        )
    )
    start = end
    pipeline.finalize(cache)  # update KV cache for the next block
generated_video = torch.cat(generated_video, dim=2)  # [B, V, T, C, H, W], range [-1, 1]
generated_num_frames = generated_video.shape[2]
print("end of streaming inference, generated_video.shape:", generated_video.shape)

# export result
if rank == 0:
    condition = hdmap_videos[:, :, :generated_num_frames]
    canvas = rearrange(
        torch.cat([condition, generated_video], dim=-2), "1 v t c h w -> t h (v w) c"
    )
    canvas = (canvas.float().cpu().numpy() + 1.0) / 2.0  # range [0, 1]
    canvas = (canvas * 255).astype(np.uint8)
    save_path = f"outputs/{CONFIG_NAME}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    media.write_video(save_path, canvas, fps=30)
    print(f"saved generated video to {save_path}")

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
