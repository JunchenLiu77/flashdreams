# Flashsim

## Instructions to run Alpadreams Inference.

```bash
# 0. request interactive node with pre-built container [IPP5 cluster as example]
# note: the container `nvcr.io/nvidian/qiwu:fast-infer-v9` is built for ARM system.
srun \
    --gpus-per-node=4 -q interactive --exclusive --nodes 1 --cpus-per-gpu 36 --pty \
    --partition=gtc_demo \
    --time=24:00:00  \
    --pty \
    --container-image=nvcr.io/nvidian/qiwu:fast-infer-v9 \
    --container-mounts=/dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels,/home:/home,/cm:/cm,/usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d \
    --container-remap-root \
    --container-mount-home \
    --container-writable \
    --container-workdir=$HOME/workspace/flashsim \
    /bin/bash

# 1. setup credentials in the file `credentials/s3_checkpoint.secret` similarly with I4:
# {
#     "aws_access_key_id": "team-sil-videogen",
#     "aws_secret_access_key": <YOUR-SIL-VIDEOGEN-PDX-KEY>,
#     "endpoint_url": "https://pdx.s8k.io",
#     "region_name": "us-east-1"
# }

# 2. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 3. (optional) setup where to cache flashsim checkpoints
export FLASHSIM_CACHE_DIR=~/.cache/flashsim # default

# 4. Run inference script. Checkpoints and example data are auto-downloaded at first run.
# - single view on single GPU
PYTHONPATH=. torchrun   --standalone   --nnodes=1   --nproc_per_node=1  \
    scripts/run_alpadreams_inference.py \
    --n_cameras 1 --total_blocks 60
# - multi view on 4 GPUs
PYTHONPATH=. torchrun   --standalone   --nnodes=1   --nproc_per_node=4  \
    scripts/run_alpadreams_inference.py \
    --n_cameras 4 --total_blocks 60
```
