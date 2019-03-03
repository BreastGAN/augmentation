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
"""Contains code for loading and preprocessing image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import resources.synthetic_data as synth_data
import resources.data.features as features


def normalize_image_np(image, img_size):
    """Rescale from range [0, 255] to [-1, 1]."""
    # Image shape should be [N,H,W] or [H,W]
    assert len(image.shape) == 3 or len(image.shape) == 2
    if len(image.shape) == 2:
        axis = None
        res_shape = (img_size[0], img_size[1], 1)
    else:
        axis = [1, 2]
        res_shape = (-1, img_size[0], img_size[1], 1)
    image = np.expand_dims(image, axis=-1)  # Add channels dimension
    image = image[:img_size[0], :img_size[1], :]
    image = np.float32(image) - image.min(axis=axis)
    val = image.max(axis=axis) / 2
    image = (image - val) / val
    image = np.reshape(image, res_shape)
    # image.shape.assert_is_compatible_with(res_shape)
    return image


def normalize_image(image, img_size):
    """Rescale from range [0, 255] to [-1, 1]."""
    # Image shape should be [N,H,W] or [H,W]
    assert len(image.shape) == 3 or len(image.shape) == 2
    if len(image.shape) == 2:
        axis = None
        res_shape = (img_size[0], img_size[1], 1)
    else:
        axis = [1, 2]
        res_shape = (-1, img_size[0], img_size[1], 1)
    image = tf.expand_dims(image, axis=-1)  # Add channels dimension
    image = tf.image.crop_to_bounding_box(image, 0, 0, img_size[0], img_size[1])
    image = tf.to_float(image) - tf.reduce_min(image, axis=axis)
    val = tf.reduce_max(image, axis=axis) / 2
    image = (image - val) / val
    image = tf.reshape(image, res_shape)
    # image.shape.assert_is_compatible_with(res_shape)
    return image


def undo_normalize_image(normalized_image):
    """Convert to a numpy array that can be read by PIL."""
    # Convert from NHWC to HWC.
    normalized_image = np.squeeze(normalized_image)
    return np.uint8(normalized_image * 127.5 + 127.5)


# def _sample_patch(image, patch_size):
#   """Crop image to square shape and resize it to `patch_size`.

#   Args:
#     image: A 3D `Tensor` of HWC format.
#     patch_size: A Python scalar.  The output image size.

#   Returns:
#     A 3D `Tensor` of HWC format which has the shape of
#     [patch_size, patch_size, 3].
#   """
#   image_shape = tf.shape(image)
#   height, width = image_shape[0], image_shape[1]
#   target_size = tf.minimum(height, width)
#   image = tf.image.resize_image_with_crop_or_pad(image, target_size,
#                                                  target_size)
#   # tf.image.resize_area only accepts 4D tensor, so expand dims first.
#   image = tf.expand_dims(image, axis=0)
#   image = tf.image.resize_images(image, [patch_size, patch_size])
#   image = tf.squeeze(image, axis=0)
#   # Force image num_channels = 3
#   image = tf.tile(image, [1, 1, tf.maximum(1, 4 - tf.shape(image)[2])])
#   image = tf.slice(image, [0, 0, 0], [patch_size, patch_size, 1])
#   return image


def _provide_custom_dataset(image_file_pattern, batch_size, shuffle=True, num_threads=1, img_size=256):
    """Provides batches of custom image data.

  Args:
    image_file_pattern: A string of glob pattern of image files.
    batch_size: The number of images in each batch.
    shuffle: Whether to shuffle the read images.  Defaults to True.
    num_threads: Number of prefetching threads.  Defaults to 1.
    img_size: Size of the image.  Defaults to 256.

  Returns:
    A float `Tensor` of shape [batch_size, img_size, img_size, 3]
    representing a batch of images.
  """
    filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(image_file_pattern), shuffle=shuffle, capacity=5 * batch_size)
    image_reader = tf.WholeFileReader()

    _, image_bytes = image_reader.read(filename_queue)
    image = tf.image.decode_image(image_bytes, channels=1)
    image_norm = normalize_image(image, (img_size, img_size))

    if shuffle:
        return tf.train.shuffle_batch([image_norm],
                                      batch_size=batch_size,
                                      num_threads=num_threads,
                                      capacity=5 * batch_size,
                                      min_after_dequeue=batch_size)
    else:
        return tf.train.batch(
                [image_norm],
                batch_size=batch_size,
                num_threads=1,  # no threads so it's deterministic
                capacity=5 * batch_size)


def provide_custom_datasets(image_file_patterns, batch_size, shuffle=True, num_threads=1, img_size=256):
    """Provides multiple batches of custom image data.

  Args:
    image_file_patterns: A list of glob patterns of image files.
    batch_size: The number of images in each batch.
    shuffle: Whether to shuffle the read images.  Defaults to True.
    num_threads: Number of prefetching threads.  Defaults to 1.
    img_size: Size of the patch to extract from the image.  Defaults to 256.

  Returns:
    A list of float `Tensor`s with the same size of `image_file_patterns`.
    Each of the `Tensor` in the list has a shape of
    [batch_size, img_size, img_size, 1] representing a batch of images.

  Raises:
    ValueError: If image_file_patterns is not a list or tuple.
  """
    if not isinstance(image_file_patterns, (list, tuple)):
        raise ValueError('`image_file_patterns` should be either list or tuple, but was {}.'.format(
                type(image_file_patterns)))
    custom_datasets = []
    for pattern in image_file_patterns:
        custom_datasets.append(
                _provide_custom_dataset(
                        pattern, batch_size=batch_size, shuffle=shuffle, num_threads=num_threads, img_size=img_size))
    return custom_datasets


def normalize_synth_image(image, img_size):
    """Rescale to [-1, 1]."""
    # 2* ((res + min(res)) / max(res)) - 1
    image = tf.to_float(image)
    maxs = tf.reduce_max(image, axis=[0, 1])
    mins = tf.reduce_min(image, axis=[0, 1])
    res = (2 * (image + mins) / maxs) - 1
    res = tf.reshape(res, (img_size[0], img_size[1], 1))
    res.shape.assert_is_compatible_with([img_size[0], img_size[1], 1])
    return res


def provide_synth_dataset(batch_size, num_threads=1, img_size=(256, 256), max_thresh=2.5):
    img_size = list(img_size)
    yield_generator = synth_data.generate_synth(size=img_size, max_thresh=max_thresh)

    def generate_synth_image():
        img1, mask1, _ = next(yield_generator)
        img2, mask2, _ = next(yield_generator)
        # Expand dims
        img_size_c = img_size + [1]
        img1, mask1 = np.reshape(img1, img_size_c), np.reshape(mask1, img_size_c)
        img2, mask2 = np.reshape(img2, img_size_c), np.reshape(mask2, img_size_c)
        # Concat mask
        h = np.concatenate([img1, mask1], axis=2)
        c = np.concatenate([img2 + mask2, mask2], axis=2)
        # print("generated healthy shape:", h.shape, " generated cancer shape:", c.shape)
        return h.astype(np.float32), c.astype(np.float32)

    healthy_img, cancer_img = tf.py_func(generate_synth_image, [], (tf.float32, tf.float32))

    img_size_c = img_size + [2]
    healthy_img = tf.reshape(healthy_img, img_size_c)
    cancer_img = tf.reshape(cancer_img, img_size_c)

    # No shuffling needed. Pictures are random anyway.
    healthy_dataset = tf.train.batch(
            [healthy_img],
            batch_size=batch_size,
            num_threads=1,  # no threads so it's deterministic
            capacity=5 * batch_size)
    cancer_dataset = tf.train.batch(
            [cancer_img],
            batch_size=batch_size,
            num_threads=1,  # no threads so it's deterministic
            capacity=5 * batch_size)
    return [healthy_dataset, cancer_dataset]


def parse_example(proto, img_size):
    features = {
            "path": tf.FixedLenFeature((), tf.string, default_value=""),
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "mask": tf.FixedLenFeature((), tf.string, default_value=""),
            "width": tf.FixedLenFeature((), tf.int64, default_value=0),
            "height": tf.FixedLenFeature((), tf.int64, default_value=0),
            "label": tf.FixedLenFeature((), tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(proto, features)

    def decode_img(img):
        img = tf.decode_raw(img, tf.float32)
        img = tf.reshape(img, img_size)
        print('final img', img.get_shape())
        return img

    path = parsed_features["path"]
    image = decode_img(parsed_features["image"])
    mask = decode_img(parsed_features["mask"])
    print('image decoded', image.get_shape())
    print('mask decoded', mask.get_shape())
    imgs = [normalize_image(img, img_size) for img in [image, mask]]
    print('normalized', imgs[0].get_shape())
    concat = tf.concat(imgs, axis=-1)
    print('concat', concat.get_shape())
    return concat, image, mask, parsed_features['label'], path


def parse_example_no_session(proto):

    def get_image(example, feature_name, suffix=''):
        w = features.example_to_int(example, 'width')
        h = features.example_to_int(example, 'height')
        return features.example_to_numpy(example, feature_name + suffix, np.float32, (h, w))

    path = features.example_to_str(proto, 'path')
    image = get_image(proto, 'image')
    mask = get_image(proto, 'mask')
    imgs = [normalize_image_np(img, img.shape) for img in [image, mask]]
    concat = np.concatenate(imgs, axis=-1)
    return concat, image, mask, features.example_to_int(proto, 'label'), path


def provide_cbis_dataset(datasets, batch_size, img_size=(256, 208), num_threads=1, max_thresh=2.5, train=True):

    def load_dataset(filename):
        dataset = tf.data.TFRecordDataset(
                filename, compression_type="GZIP", num_parallel_reads=num_threads, buffer_size=buffer_size)
        dataset = dataset.map(lambda x: parse_example(x, img_size)[0], num_parallel_calls=num_threads)
        if train:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size, count=None, seed=42))
        dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features

    buffer_size = 100
    shuffle_buffer_size = 100

    return [load_dataset(x) for x in datasets]
