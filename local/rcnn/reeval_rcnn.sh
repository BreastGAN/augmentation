#!/bin/bash

set -e

output_dir="$1"

echo "Output dir: $output_dir"

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
exec python -m models.rcnn.reevaluate "$output_dir"
