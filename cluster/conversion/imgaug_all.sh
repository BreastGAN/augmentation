#!/bin/bash

set -e

dataserver="biwidl100"
bd="/scratch_net/$dataserver/$USER"
transformed="$bd/transformed_256x204/small_all_256x204_final"
out_folder="$bd/mammography/data_out/imgaug"
mkdir -p "$out_folder"

datasets=( "train" "eval" "test" )
cancers=( "healthy" "cancer" )
for dataset in "${datasets[@]}"; do
    for cancer in "${cancers[@]}"; do
        qsub imgaug.sh "$transformed/${cancer}.${dataset}.tfrecord" "$out_folder/${cancer}.${dataset}_gen.tfrecord"
    done
done
