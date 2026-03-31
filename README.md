# Flashsim

## Instructions to run Alpadreams Inference.

```bash
# 1. Download sampled data from S3 bucket.
aws s3 sync \
    s3://flashsim/assets/example_data/alpadreams \
    ./assets/example_data/alpadreams \
    --profile team-sil-videogen \
    --endpoint-url https://pdx.s8k.io

# 2. Run inference demo. Checkpoints are auto-downloaded at first run.
pip install -e .
python scripts/run_alpadreams_inference.py 
```