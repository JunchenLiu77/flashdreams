#!/bin/bash

CURRENT_DIR=$(dirname $(realpath $0))

aws s3 sync s3://flashdreams/assets/checkpoints $CURRENT_DIR/checkpoints --profile team-sil-videogen --endpoint-url https://pdx.s8k.io
aws s3 sync s3://flashdreams/assets/example_data $CURRENT_DIR/example_data --profile team-sil-videogen --endpoint-url https://pdx.s8k.io
