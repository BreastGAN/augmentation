#!/bin/bash

folder_name="$1"
out_folder_name="$2"
echo "Transformed folder name: $folder_name"
echo "Transformed out folder name: $out_folder_name"

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
exec python -m notebooks.image_conversion --merge "True" --height 256 --width 204 --in_folder="$folder_name" --out_folder="$out_folder_name"
