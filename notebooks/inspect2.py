# Copyright 2019 Lukas Jendele and Ondrej Skopek.
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

import numpy as np
import tensorflow as tf

import resources.image_utils as imutils

img_size = 256


def normalize_image(image, img_size):
    """Rescale from range [0, 255] to [-1, 1]."""
    res = tf.image.resize_image_with_crop_or_pad(image, target_height=img_size, target_width=img_size)
    res = (tf.to_float(res) - 127.5) / 127.5
    res = tf.reshape(res, (img_size, img_size, 1))
    res.shape.assert_is_compatible_with([img_size, img_size, 1])
    return res


def to_examples(proto):
    features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "mask": tf.FixedLenFeature((), tf.string, default_value=""),
            "width": tf.FixedLenFeature((), tf.int64, default_value=0),
            "height": tf.FixedLenFeature((), tf.int64, default_value=0),
            "label": tf.FixedLenFeature((), tf.int64, default_value=0),
    }
    parsed_features = tf.parse_single_example(proto, features)

    def decode_img(img):
        img = tf.decode_raw(img, tf.float32)
        height, width = parsed_features['height'], parsed_features['width']
        img = tf.reshape(img, tf.stack([height, width, 1]))
        return img

    image = decode_img(parsed_features["image"])
    mask = decode_img(parsed_features["mask"])
    # print('image decoded', image.get_shape())
    # print('mask decoded', mask.get_shape())
    """
    # Resize
    method = tf.image.ResizeMethod.BILINEAR
    imgs = [tf.image.resize_images(img, (img_size, img_size), method=method) for img in [image, mask]]
    imgs = [normalize_image(img, img_size) for img in imgs]
    print('normalized', imgs[0].get_shape())
    concat = tf.concat(imgs, axis=-1)
    print('concat', concat.get_shape())
    """
    return image, mask


def show_images(file):
    compress = tf.python_io.TFRecordCompressionType.GZIP
    options = tf.python_io.TFRecordOptions(compress)
    with tf.Session() as session:
        iterr = tf.python_io.tf_record_iterator(file, options)
        for example in iterr:
            img, mask = to_examples(example)
            img, mask = session.run([img, mask])
            assert img.shape == mask.shape
            print('shape', img.shape)
            img = np.reshape(img, img.shape[:-1])
            mask = np.reshape(mask, mask.shape[:-1])
            imutils.show_img(img)
            imutils.show_img(mask)
