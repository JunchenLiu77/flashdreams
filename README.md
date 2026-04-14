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
cat > credentials/s3_checkpoint.secret.2 <<EOF
{
  "aws_access_key_id": "team-sil-videogen",
  "aws_secret_access_key": <YOUR-SIL-VIDEOGEN-PDX-KEY>,
  "endpoint_url": "https://pdx.s8k.io",
  "region_name": "us-east-1"
}
EOF

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



## Instructions to run Self-forcing T2V Inference.

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
PYTHONPATH=. torchrun   --standalone   --nnodes=1   --nproc_per_node=1 \
    scripts/run_wan_t2v.py \
    --total_blocks 7
```


## Instructions to run Causal-forcing T2V and I2V Inference.

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
# - T2V
PYTHONPATH=. torchrun   --standalone   --nnodes=1   --nproc_per_node=1 \
    scripts/run_wan_t2v.py \
    --total_blocks 21 \
    --overwrite_config_name casual_forcing_framewise

# - I2V
PYTHONPATH=. torchrun   --standalone   --nnodes=1   --nproc_per_node=1 \
    scripts/run_wan_t2v.py \
    --total_blocks 21 \
    --overwrite_config_name casual_forcing_framewise \
    --prompt_or_txt_path assets/example_data/i2v/prompt.txt  \
    --image_path assets/example_data/i2v/image.jpg
```


## Instructions to run FastVideo Wan2.2 Causal T2V and I2V Inference.
reference: [FastVideo official inference script](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py)

```bash
# 0. request interactive node with pre-built container save as above alpadreams demo.

# 1. setup huggingface
# - (required) huggingface token
export HF_TOKEN=<YOUR-HF-TOKEN>
# - (optional) huggingface cache path
export HF_HOME=~/.cache/huggingface # default

# 2. Run inference script. Checkpoint will be auto-downloaded at first run from huggingface.
# - T2V
PYTHONPATH=. torchrun   --standalone   --nnodes=1   --nproc_per_node=1 \
    scripts/run_wan2_2_i2v.py \
    --total_blocks 21

# # - I2V (not supported yet)
# PYTHONPATH=. torchrun   --standalone   --nnodes=1   --nproc_per_node=1 \
#     scripts/run_wan2_2_i2v.py \
#     --total_blocks 21 \
#     --prompt_or_txt_path assets/example_data/i2v/prompt.txt  \
#     --image_path assets/example_data/i2v/image.jpg
```
