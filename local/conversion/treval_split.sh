#!/bin/bash

set -e

size="256x204"

src_dir="${1:-./data_in/transformed/traineval_$size}"
dst_dir="${2:-./data_in/transformed/small_all_${size}_final}"

healthy="$src_dir/healthy.tfrecord"
cancer="$src_dir/cancer.tfrecord"

test_src_dir="${1:-./data_in/transformed/test_$size}"
cp "$test_src_dir/healthy.tfrecord" "$dst_dir/healthy.test.tfrecord"
cp "$test_src_dir/cancer.tfrecord" "$dst_dir/cancer.test.tfrecord"

PROJECT_HOME="."
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Run model:"
cd $PROJECT_HOME
exec python -m notebooks.treval_split --healthy "$healthy" --cancer "$cancer" --out_dir="$dst_dir" --train_percent 0.99

