#!/bin/bash
set -e

dataserver="biwidl100"

traineval_data=( bcdr01 bcdr02 ) #zrh )
#traineval_data=( bcdr01 bcdr02 cbis ) #zrh )
#traineval_data=( )
test_data=( inbreast )

height="${1:-256}"
width="${2:-204}"
user="${3:-$USER}"

size="${height}x${width}"
transformed="/scratch_net/$dataserver/$user/transformed"

traineval_folder="${transformed}/traineval_${size}"
mkdir -p "$traineval_folder"
for dataset in ${traineval_data[@]}; do
    echo "TrainEval: $dataset"
    dataset_folder="$transformed/${dataset}_${size}"
    cp "$dataset_folder/cancer.tfrecord" "$traineval_folder/label_1.${dataset}.tfrecord"
    cp "$dataset_folder/healthy.tfrecord" "$traineval_folder/label_0.${dataset}.tfrecord"
done
qsub merge_images.sh "$traineval_folder" "$traineval_folder"

test_folder="${transformed}/test_${size}"
mkdir -p "$test_folder"
for dataset in ${test_data[@]}; do
    echo "TrainEval: $dataset"
    dataset_folder="$transformed/${dataset}_${size}"
    cp "$dataset_folder/cancer.tfrecord" "$test_folder/label_1.${dataset}.tfrecord"
    cp "$dataset_folder/healthy.tfrecord" "$test_folder/label_0.${dataset}.tfrecord"
done
qsub merge_images.sh "$test_folder" "$test_folder"

