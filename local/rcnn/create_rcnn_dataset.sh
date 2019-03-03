#!/bin/bash

tfrecords="$1"
output_dir="$2"
convert_generated="$3"
flip_label_gen="$4"
inbreast_corrections="$5"

echo "Dataset: $tfrecords"
echo "Output dir: $output_dir"

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
exec python -m resources.data.create_rcnn_dataset --input_file_pattern "$tfrecords" --output_file "$output_dir" $convert_generated $flip_label_gen $inbreast_corrections
