#!/bin/bash

set -e

exp="NoAugment_ICNR_nnUPSAMPLE_Lam0.0_SpectralNorm"
mask_chkpt="40000"
nomask_chkpt="25000"

in_data="data_in/transformed/small_all_512x408_final"
data="data_out"
r_data="$data/rcnn"

inbreast_corrections="--inbreast_corrections=./resources/data/inbreast_ground_truth_cancer_bbox_rois.tsv"

rm -rf "$r_data"
mkdir -p "$r_data"
mkdir -p "$r_data/orig_mask"
mkdir -p "$r_data/gan_mask"
mkdir -p "$r_data/gan_nomask"
mkdir -p "$r_data/aug_mask"
for stage in "train" "eval" "test"; do
    for label in "healthy" "cancer"; do
        # Copy original data
        cp "$in_data/${label}.${stage}.tfrecord" "$r_data/orig_mask/"
        # Classically augmented data
        cp "$data/imgaug/${label}.${stage}_gen.tfrecord" "$r_data/aug_mask/${label}.${stage}.tfrecord"
    done
    for label in "H2C" "C2H"; do
        # Mask true inferred data
        cp "$data/MaskTrue_${exp}_${mask_chkpt}_steps_inference_${stage}/${label}_gen.tfrecord" "$r_data/gan_mask/${label}.${stage}.tfrecord"
        # Mask false inferred data
        cp "$data/MaskFalse_${exp}_${nomask_chkpt}_steps_inference_${stage}/${label}_gen.tfrecord" "$r_data/gan_nomask/${label}.${stage}.tfrecord"
    done

    if [ $stage == "test" ]; then
        test_corrections="$inbreast_corrections"
    else
        test_corrections=""
    fi

    if [ $stage == "train" ]; then
        train_flip="--flip_label_gen"
    else
        train_flip=""
    fi

    ./local/rcnn/create_rcnn_dataset.sh "$r_data/orig_mask/*.${stage}.tfrecord" "$r_data/orig_mask_rcnn/${stage}.tfrecord" "" "" "$test_corrections"
    ./local/rcnn/create_rcnn_dataset.sh "$r_data/gan_mask/*.${stage}.tfrecord" "$r_data/gan_mask_rcnn/${stage}.tfrecord" "--convert_generated" "$train_flip" "$test_corrections"
    ./local/rcnn/create_rcnn_dataset.sh "$r_data/gan_nomask/*.${stage}.tfrecord" "$r_data/gan_nomask_rcnn/${stage}.tfrecord" "--convert_generated" "$train_flip" "$test_corrections"
    ./local/rcnn/create_rcnn_dataset.sh "$r_data/aug_mask/*.${stage}.tfrecord" "$r_data/aug_mask_rcnn/${stage}.tfrecord" "--convert_generated" "" "$test_corrections"
done
