#!/bin/bash

dataset="$1"
cancer="$2"

echo "Dataset: $dataset"
echo "Cancer: $cancer"

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
exec python -m notebooks.image_conversion --cancer "$cancer" --height 256 --width 204 --merge "False" --dataset "$dataset" --out_folder="$PROJECT_HOME/transformed/small_all_256x204" --in_folder="$PROJECT_HOME/data_in"
