import os
import argparse

import torch
import numpy as np
import mediapy as media
from einops import rearrange
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
    "--prompt", type=str, default=DEFAULT_TEXT_PROMPT, help="Text prompt."
)
args = parser.parse_args()

CAMERA_NAMES = ["default"]
DATA = [
    {
        "prompt": args.prompt,
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
for data in DATA:  # loop over views
    prompts.append(data["prompt"])
prompts = [prompts]  # add a batch dimension

# initialize pipeline
pipeline = WAN2_1_CONFIGS[CONFIG_NAME].setup(device=device)
cache = pipeline.initialize_cache(video_height=480, video_width=832, text=prompts)

torch.cuda.synchronize()
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# streaming inference
start = 0
generated_video = []
for i in range(args.total_blocks):
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
