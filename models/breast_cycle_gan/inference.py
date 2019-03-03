# Copyright 2019 Lukas Jendele and Ondrej Skopek.
# Adapted from The TensorFlow Authors, under the ASL 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools

import numpy as np
import tensorflow as tf

import models.breast_cycle_gan.data_provider as data_provider
from models.breast_cycle_gan.train import _define_model
import resources.data.features as features
from resources.data.transformer import OPTIONS

flags = tf.flags
tfgan = tf.contrib.gan

flags.DEFINE_string('checkpoint_path', '', 'CycleGAN checkpoint path created by train.py.'
                    '(e.g. "/mylogdir/model.ckpt-18442")')

flags.DEFINE_string('generated_dir', '/scratch_net/biwidl104/jendelel/mammography/data_out/generated/',
                    'Where to output the generated images.')

FLAGS = flags.FLAGS


def _make_dir_if_not_exists(dir_path):
    """Make a directory if it does not exist."""
    if not tf.gfile.Exists(dir_path):
        tf.gfile.MakeDirs(dir_path)


def normalize_back(img):
    """Normalizes back between -1 and 1"""
    if np.sum(img) == 0:
        # Avoid dividing by zero.
        return img
    # print("before: ", np.min(img), np.max(img))
    img -= np.min(img)
    img /= np.max(img)
    img = (img * 2) - 1
    # print("after: ", np.min(img), np.max(img))
    return img


def to_example_extended(img_path, img_orig, img_gen, mask_orig, mask_gen, width, height, label_orig, label_gen):
    assert img_orig.shape == mask_orig.shape
    assert img_orig.shape == img_gen.shape
    assert img_gen.shape == mask_gen.shape
    assert isinstance(label_orig, np.int64)
    assert isinstance(label_gen, np.int64)
    features_dict = features.to_feature_dict(img_path, img_orig, mask_orig, width, height, label_orig)
    # Add the generated data with the _gen suffix to the dict.
    features_dict.update(features.to_feature_dict(img_path, img_gen, mask_gen, width, height, label_gen, suffix="_gen"))
    # Create an example protocol buffer
    return features.to_example(features_dict)


def export_images(cyclegan_model, include_masks, sess):

    def export_image_batch(filename_fn, gan_model, label, reconstructions, writer):
        image_list = [
                gan_model.generator_inputs[:, :, :, 0], gan_model.generated_data[:, :, :, 0],
                reconstructions[:, :, :, 0]
        ]
        fetches = image_list

        if include_masks:
            image_list_masks = [
                    gan_model.generator_inputs[:, :, :, 1], gan_model.generated_data[:, :, :, 1],
                    reconstructions[:, :, :, 1]
            ]
            fetches.extend(image_list_masks)

        outputs = sess.run(fetches)

        if include_masks:
            input_img, output_img, reconstruction_img, input_mask, output_mask, reconstruction_mask = outputs
            output_img, output_mask = normalize_back(output_img), normalize_back(output_mask)
        else:
            input_img, output_img, reconstruction_img = outputs
            input_mask, output_mask, reconstruction_mask = np.zeros_like(input_img), np.zeros_like(
                    output_img), np.zeros_like(reconstruction_img)

        reconstruction_mask[0]  # dummy line to shut up flake8.
        for i in range(input_img.shape[0]):
            proto = to_example_extended(
                    filename_fn(i), input_img[i], output_img[i], input_mask[i], output_mask[i], FLAGS.width,
                    FLAGS.height, label, np.int64(not bool(label)))
            writer.write(proto.SerializeToString())

    def filename_fn(sample, direction, batch):
        return "img_{}_{}".format(direction, FLAGS.batch_size * batch + sample)

    def output_filename(fname):
        _, file_name = os.path.split(fname)
        file_name = os.path.splitext(file_name)[0]
        return os.path.join(FLAGS.generated_dir, file_name + '_gen.tfrecord')

    with tf.python_io.TFRecordWriter(output_filename(FLAGS.image_x_file), options=OPTIONS) as writer:
        try:
            print("Converting data H2C")
            batch = 0
            while True:
                print("Batch:", batch)
                batch += 1
                export_image_batch(
                        functools.partial(filename_fn, direction="H2C", batch=batch), cyclegan_model.model_x2y,
                        np.int64(0), cyclegan_model.reconstructed_x, writer)
        except tf.errors.OutOfRangeError:
            print("Enf of dataset H2C!")

    with tf.python_io.TFRecordWriter(output_filename(FLAGS.image_y_file), options=OPTIONS) as writer:
        try:
            print("Converting data C2H")
            batch = 1
            while True:
                print("Batch:", batch)
                batch += 1
                export_image_batch(
                        functools.partial(filename_fn, direction="C2H", batch=batch), cyclegan_model.model_y2x,
                        np.int64(1), cyclegan_model.reconstructed_y, writer)
        except tf.errors.OutOfRangeError:
            print("Enf of dataset C2H!")


def main(_):
    if FLAGS.generated_dir:
        _make_dir_if_not_exists(FLAGS.generated_dir)

    with tf.Session() as sess:
        with tf.name_scope('inputs'):
            if FLAGS.use_dataset == 'synth':
                # Generated two channels. First channel image, second channel is mask.
                images_healthy, images_cancer = data_provider.provide_synth_dataset(
                        batch_size=FLAGS.batch_size, img_size=(FLAGS.height, FLAGS.width))
            elif FLAGS.use_dataset == 'cbis':
                # Generated two channels. First channel image, second channel is mask.
                images_healthy, images_cancer = data_provider.provide_cbis_dataset(
                        [FLAGS.image_x_file, FLAGS.image_y_file],
                        batch_size=FLAGS.batch_size,
                        img_size=(FLAGS.height, FLAGS.width),
                        train=False)
            else:
                images_healthy, images_cancer = data_provider.provide_custom_datasets(
                        [FLAGS.image_set_x_file_pattern, FLAGS.image_set_y_file_pattern],
                        batch_size=FLAGS.batch_size,
                        img_size=(FLAGS.height, FLAGS.width))

        # Define CycleGAN model.
        print("images healthy", images_healthy.get_shape())
        print("images cancer", images_cancer.get_shape())
        cyclegan_model = _define_model(images_healthy, images_cancer, FLAGS.include_masks)

        print("Restoring from", FLAGS.checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_path)

        export_images(cyclegan_model, FLAGS.include_masks, sess)


if __name__ == '__main__':
    tf.app.run()
