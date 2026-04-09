import os
import argparse

import torch
import numpy as np
import mediapy as media
from einops import rearrange
import cv2
from huggingface_hub import login as huggingface_login

from flashsim.distributed import init as distributed_init
from flashsim.configs.wan2_1 import WAN2_1_CONFIGS
from flashsim.pipeline.wan2_1 import ProfileEvents

DEFAULT_TEXT_PROMPT = "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."

parser = argparse.ArgumentParser()
parser.add_argument(
    "--total_blocks", type=int, default=60, help="Total blocks to generate."
)
parser.add_argument(
    "--overwrite_config_name", type=str, default=None, help="Overwrite config name."
)
parser.add_argument(
    "--prompt_or_txt_path",
    type=str,
    default=DEFAULT_TEXT_PROMPT,
    help="Text prompt or text file path.",
)
parser.add_argument("--image_path", type=str, default=None, help="Image path.")
parser.add_argument("--video_height", type=int, default=480, help="Video height.")
parser.add_argument("--video_width", type=int, default=832, help="Video width.")
args = parser.parse_args()

if args.prompt_or_txt_path.endswith(".txt"):
    with open(args.prompt_or_txt_path, "r") as f:
        args.prompt_or_txt_path = f.readlines()[0]

CAMERA_NAMES = ["default"]
DATA = [
    {
        "first_frame_path": args.image_path,
        "prompt": args.prompt_or_txt_path,
    }
    for name in CAMERA_NAMES
]
CONFIG_NAME = "self_forcing"

if args.overwrite_config_name is not None:
    CONFIG_NAME = args.overwrite_config_name
print(f"Running Wan2_1 inference with config: {CONFIG_NAME}")

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
prompts = []
first_frames: list[torch.Tensor] | None = []
for data in DATA:  # loop over views
    prompts.append(data["prompt"])
    first_frame_path = data["first_frame_path"]
    if first_frame_path is not None:
        first_frame = media.read_image(first_frame_path)[
            ..., :3
        ]  # only keep RGB channels
        first_frame = cv2.resize(first_frame, (args.video_width, args.video_height))
        first_frame = (
            torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
        )  # range [-1, 1]
        first_frame = rearrange(first_frame, "h w c -> 1 c h w")  # [1, C, H, W]
        first_frames.append(first_frame)
# add a batch dimension
prompts = [prompts]
if len(first_frames) > 0:
    first_frames = torch.stack(first_frames, dim=0).unsqueeze(0)  # [B, V, 1, C, H, W]
    start_index = 1
else:
    first_frames = None
    start_index = 0

# initialize pipeline
pipeline = WAN2_1_CONFIGS[CONFIG_NAME].setup(device=device)
cache = pipeline.initialize_cache(
    video_height=args.video_height,
    video_width=args.video_width,
    text=prompts,
    image=first_frames,
)

torch.cuda.synchronize()
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# streaming inference
start = 1 if first_frames is not None else 0
generated_video = [first_frames] if first_frames is not None else []
for i in range(start_index, args.total_blocks):
    num_frames = pipeline.get_num_frames(i)
    end = start + num_frames
    print(
        f"autoregressive_index: {i}, num_frames: {num_frames}, start: {start}, end: {end}"
    )
    generated_video.append(
        pipeline.streaming_inference(
            autoregressive_index=i,
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
