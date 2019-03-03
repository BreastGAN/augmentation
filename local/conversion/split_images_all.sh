#!/bin/bash
set -e

traineval_data=( bcdr01 bcdr02 ) #zrh )
test_data=( inbreast )

height="${1:-256}"
width="${2:-204}"

size="${height}x${width}"
transformed="./data_in/transformed"

traineval_folder="${transformed}/traineval_${size}"
mkdir -p "$traineval_folder"
for dataset in ${traineval_data[@]}; do
    echo "TrainEval: $dataset"
    dataset_folder="$transformed/${dataset}_${size}"
    cp "$dataset_folder/cancer.tfrecord" "$traineval_folder/label_1.${dataset}.tfrecord"
    cp "$dataset_folder/healthy.tfrecord" "$traineval_folder/label_0.${dataset}.tfrecord"
done
./local/merge_images.sh "$traineval_folder" "$traineval_folder"

test_folder="${transformed}/test_${size}"
mkdir -p "$test_folder"
for dataset in ${test_data[@]}; do
    echo "TrainEval: $dataset"
    dataset_folder="$transformed/${dataset}_${size}"
    cp "$dataset_folder/cancer.tfrecord" "$test_folder/label_1.${dataset}.tfrecord"
    cp "$dataset_folder/healthy.tfrecord" "$test_folder/label_0.${dataset}.tfrecord"
done
./local/merge_images.sh "$test_folder" "$test_folder"

