#!/bin/bash

set -e

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

checkpoint_path="${1}"
masks="${2:-True}"
icnr="${3:-False}"
upsample_method="${4:-conv2d_transpose}"
loss_identity_lambda="${5:-0.0}"
output_dir="${6}"
split="${7:-train}"
spectral_norm="${8:-True}"

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
base_path="$PROJECT_HOME/data_in/transformed/small_all_256x204_final"
exec python -m models.breast_cycle_gan.inference --height 256 --width 204 --image_x_file "$base_path/healthy.$split.tfrecord" --image_y_file "$base_path/cancer.$split.tfrecord" --include_masks="$masks" --checkpoint_path="$checkpoint_path" --use_icnr="$icnr" --upsample_method="$upsample_method" --loss_identity_lambda="$loss_identity_lambda" --use_spectral_norm="$spectral_norm" --generated_dir="$output_dir" --batch_size=5

