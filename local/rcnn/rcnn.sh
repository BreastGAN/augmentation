#!/bin/bash

train_f="$1"
eval_f="$2"
test_f="$3"
log_dir="$4"

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
exec python -m models.rcnn.train --config DATA.TRAIN_PATTERN="$train_f" DATA.VAL_PATTERN="$eval_f" DATA.TEST_PATTERN="$test_f" MODE_MASK=False BACKBONE.WEIGHTS="$PROJECT_HOME/data_in/ImageNet-ResNet50.npz" --logdir "${log_dir}" # --load data_out/FRCNN_split_gen_bak/model-45000
