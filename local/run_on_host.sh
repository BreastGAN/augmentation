#!/bin/bash

set -e

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

timestamp="$(date '+%Y_%m_%d-%H_%M_%S_%s')"
exp_name="${1:-BreastCycleGan-$timestamp}"
masks="${2:-True}"
icnr="${3:-False}"
upsample_method="${4:-conv2d_transpose}"
loss_identity_lambda="${5:-0.0}"
checkpoint_hook_steps="${6:-25000}"
spectral_norm="${7:-False}"

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
base_path="$PROJECT_HOME/data_in/transformed/small_all_256x204_final"
exec python -m models.breast_cycle_gan.train --height 256 --width 204 --image_x_file "$base_path"/healthy.train.tfrecord --image_y_file "$base_path"/cancer.train.tfrecord --include_masks="$masks" --train_log_dir="data_out/$exp_name" --use_icnr="$icnr" --upsample_method="$upsample_method" --loss_identity_lambda="$loss_identity_lambda" --checkpoint_hook_steps="$checkpoint_hook_steps" --use_spectral_norm="$spectral_norm"
