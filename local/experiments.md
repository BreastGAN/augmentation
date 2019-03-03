== Image conversion

Folder: `conversion/`
Always proceed after all jobs finished.

1. `./convert_images_all.sh`
2. `./merge_images_all.sh`
3. `./split_images_all.sh`
4. `qsub ./treval_split.sh`
5. `./imgaug_all.sh`

Then change the path to `.../transformed_256x204/...` and run everything again,
but add 'cbis' to the list of training datasets in `split_images_all.sh`.

== GAN pre-training

Folder: `gan/`

1. `./train.sh`. Wait 24 hours.
2. `./infer.sh`
3. Optional: `./to_png.sh`

== RCNN training

Folder: `rcnn/`

=== RCNN datasets

1. Fill in the checkpoints and run: `./create_rcnn_datasets.sh`
2. Run `./setup/download_resnet.sh` and then `./train_rcnns.sh`
3. Fill in the checkpoints and run: `./eval_rcnns.sh`

Results and plots are in `data_out/rcnns/`.

