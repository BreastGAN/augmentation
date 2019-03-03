#!/bin/bash

r_data="./data_out/rcnn"

# Experiment 1: Original training set, GAN-augmented eval set
./local/rcnn/rcnn.sh "$r_data/orig_mask_rcnn/train.tfrecord" "$r_data/gan_mask_rcnn/eval.tfrecord" "$r_data/gan_mask_rcnn/test.tfrecord" "$r_data/data_out/orig_mask_rcnn"
# Experiment 2: GAN-aumgented training set, GAN-augmented eval set
./local/rcnn/rcnn.sh "$r_data/gan_mask_rcnn/train.tfrecord" "$r_data/gan_mask_rcnn/eval.tfrecord" "$r_data/gan_mask_rcnn/test.tfrecord" "$r_data/data_out/gan_mask_rcnn"
# Experiment 3: imgaug-aumgented training set, GAN-augmented eval set
./local/rcnn/rcnn.sh "$r_data/aug_mask_rcnn/train.tfrecord" "$r_data/gan_mask_rcnn/eval.tfrecord" "$r_data/gan_mask_rcnn/test.tfrecord" "$r_data/data_out/aug_mask_rcnn"
