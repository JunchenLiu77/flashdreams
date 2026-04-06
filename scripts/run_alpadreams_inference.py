import os

import torch
import numpy as np
import mediapy as media
from einops import rearrange
from huggingface_hub import login as huggingface_login

from flashsim.configs.alpadreams import ALPADREAMS_CONFIGS
from flashsim.io.s3_sync import sync_s3_dir_to_local

EXAMPLE_DATA_DIR_S3 = "s3://flashsim/assets/example_data/alpadreams"
EXAMPLE_DATA_DIR_LOCAL = os.path.join(
    os.path.dirname(__file__), "../assets/example_data/alpadreams"
)
HDMAP_VIDEO_PATH = os.path.join(EXAMPLE_DATA_DIR_LOCAL, "camera_front_wide_120fov.mp4")
FIRST_FRAME_PATH = os.path.join(EXAMPLE_DATA_DIR_LOCAL, "camera_front_wide_120fov.png")
PROMPT = (
    "Driving scene from a front-facing car camera. Urban environment with roads, vehicles, pedestrians, "
    "traffic signs, and buildings. Clear visibility, realistic lighting, photorealistic quality. "
    "High resolution dashcam footage of city driving."
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

device = torch.device("cuda")
dtype = torch.bfloat16

# prepare data
first_frame = media.read_image(FIRST_FRAME_PATH)
first_frame = (
    torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
)  # range [-1, 1]
first_frame = rearrange(first_frame, "h w c -> 1 1 1 c h w")  # [B, V, 1, C, H, W]
hdmap_video = media.read_video(HDMAP_VIDEO_PATH)
hdmap_video = (
    torch.from_numpy(hdmap_video).to(dtype=dtype, device=device) / 127.5 - 1.0
)  # range [-1, 1]
hdmap_video = rearrange(hdmap_video, "t h w c -> 1 1 t c h w")  # [B, V, T, C, H, W]
batch_size, num_views, hdmap_num_frames, _3, height, width = hdmap_video.shape
text = [[PROMPT]]  # [B, V]
print("loaded hdmap_video.shape:", hdmap_video.shape)

# initialize pipeline
pipeline = ALPADREAMS_CONFIGS["sv_2steps_chunk3_loc6_vae_vae"].setup()
cache = pipeline.initialize_cache(text=text, image=first_frame)

# streaming inference
start = 0
generated_video = []
for i in range(60):
    num_frames = pipeline.get_num_frames(i)
    end = start + num_frames
    if end > hdmap_num_frames:
        break
    print(
        f"autoregressive_index: {i}, num_frames: {num_frames}, start: {start}, end: {end}"
    )
    generated_video.append(
        pipeline.streaming_inference(
            autoregressive_index=i, hdmap=hdmap_video[:, :, start:end], cache=cache
        )
    )
    start = end
    pipeline.finalize(cache)  # update KV cache for the next block
generated_video = torch.cat(generated_video, dim=2)  # [B, V, T, C, H, W], range [-1, 1]
generated_num_frames = generated_video.shape[2]
print("end of streaming inference, generated_video.shape:", generated_video.shape)

# export result
condition = hdmap_video[:, :, :generated_num_frames]
canvas = rearrange(
    torch.cat([condition, generated_video], dim=-2), "1 1 t c h w -> t h w c"
)
canvas = (canvas.float().cpu().numpy() + 1.0) / 2.0  # range [0, 1]
canvas = (canvas * 255).astype(np.uint8)
save_path = "outputs/generated_video.mp4"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
media.write_video(save_path, canvas, fps=30)
print(f"saved generated video to {save_path}")
