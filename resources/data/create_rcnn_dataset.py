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

import os
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import cm

from resources.data.features import numpy_to_feature, str_to_feature, int_to_feature
from resources.data.features import example_to_int, example_to_str, example_to_numpy
import resources.image_utils as imutils

import skimage as ski
# These need to be imported explicitely.
import skimage.morphology  # noqa: F401
import skimage.measure  # noqa: F401
import skimage.segmentation  # noqa: F401
import skimage.filters  # noqa: F401

OPTIONS = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)


def get_image(example, feature_name, suffix=''):
    w = example_to_int(example, 'width')
    h = example_to_int(example, 'height')
    return example_to_numpy(example, feature_name + suffix, np.float32, (h, w))


def get_examples(tfrecords_glob, options=OPTIONS):
    for file in tf.gfile.Glob(tfrecords_glob):
        print("Processing file {}".format(file))
        for record in tf.python_io.tf_record_iterator(file, options=options):
            example = tf.train.Example()
            example.ParseFromString(record)
            yield example


def to_png(img, path):
    img = imutils.normalize(img.copy(), new_min=0, new_max=255)
    matplotlib.image.imsave(path, img, cmap=cm.gray)


def feature_dict(img_path, old_path, bboxes, width, height, label, suffix=""):
    assert isinstance(label, np.int64)
    feature = {
            'path' + suffix: str_to_feature(old_path),
            'image_path' + suffix: str_to_feature(img_path),
            'bboxes' + suffix: numpy_to_feature(bboxes, np.float32),
            'width' + suffix: int_to_feature(width),
            'height' + suffix: int_to_feature(height),
            'label' + suffix: int_to_feature(label)
    }
    return feature


def to_example(feature_dict):
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def mask_to_boxes(mask_image):
    if np.sum(mask_image) == 0:
        return np.empty([0, 4], dtype=np.float32)
    thresh = ski.filters.threshold_otsu(mask_image)
    bw = ski.morphology.closing(mask_image > thresh, ski.morphology.square(3))

    # Remove artifacts connected to image border
    cleared = ski.segmentation.clear_border(bw)

    lbl = ski.measure.label(cleared)
    props = ski.measure.regionprops(lbl)
    bboxes = []
    width = mask_image.shape[1]
    height = mask_image.shape[0]
    for prop in props:
        if prop.area > 10:
            y1, x1, y2, x2 = prop.bbox
            x1 = np.clip(float(x1), 0, width)
            x2 = np.clip(float(x2), 0, width)
            y1 = np.clip(float(y1), 0, height)
            y2 = np.clip(float(y2), 0, height)
            bboxes.append([x1, y1, x2, y2])
    return np.asarray(bboxes, dtype='float32')


def inbreast_label(img_path, lbl, inbreast_corrections):
    if not inbreast_corrections:
        return lbl
    for malignant_file in inbreast_corrections:
        if malignant_file in img_path:
            print("Correcting file {}".format(img_path))
            return 1
    return 0


def create_new_example(example, inbreast_corrections=None, suffix='', flip_label_gen=False):
    # Save image as png
    image = get_image(example, 'image', suffix=suffix)
    old_path = example_to_str(example, 'path')
    old_path_with_suffix = old_path + suffix
    new_path = os.path.join(img_dir, old_path_with_suffix.replace('/', '_')) + '.png'
    to_png(image, new_path)

    mask = get_image(example, 'mask', suffix)
    old_mask_path = example_to_str(example, 'path') + suffix + '_mask'
    new_mask_path = os.path.join(img_dir, old_mask_path.replace('/', '_')) + '.png'
    to_png(mask, new_mask_path)

    # Convert mask to bboxes
    bboxes = mask_to_boxes(mask)
    lbl = example_to_int(example, 'label')
    lbl = inbreast_label(old_path, lbl, inbreast_corrections)
    if suffix and flip_label_gen:
        print("Swapping label for image: {}".format(old_path_with_suffix))
        lbl = int(not bool(lbl))  # Swap the label
    # Save it all into Example proto
    features = feature_dict(
            img_path=new_path,
            old_path=old_path_with_suffix,
            bboxes=bboxes,
            width=np.int64(image.shape[1]),
            height=np.int64(image.shape[0]),
            label=np.int64(lbl))
    return to_example(features)


def convert(examples, img_dir, include_generated=False, inbreast_corrections=None, flip_label_gen=False):
    os.makedirs(img_dir, exist_ok=True)
    for example in examples:
        yield create_new_example(example, inbreast_corrections)
        if include_generated:
            yield create_new_example(example, inbreast_corrections, "_gen", flip_label_gen=flip_label_gen)


def save_examples(examples, fname):
    with tf.python_io.TFRecordWriter(fname, options=OPTIONS) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def load_inbreast_corrections(inbreast_corrections_path):
    if not inbreast_corrections_path:
        return []
    res = []
    with open(inbreast_corrections_path, 'r') as f:
        fiter = iter(f)
        next(fiter)
        for line in fiter:
            fname = line.split("\t")[0]
            res.append(fname)
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_pattern", help="Global pattern for input tfrecords.")
    parser.add_argument("--output_file", help="Output file path.")
    parser.add_argument("--inbreast_corrections", help="Path to inbreast corrections.")
    parser.add_argument("--convert_generated", help="Whether to convert generated images as well.", action='store_true')
    parser.add_argument("--flip_label_gen", help="Whether to flip label for  generated images.", action='store_true')
    args = parser.parse_args()
    assert args.input_file_pattern and args.output_file

    tfrecords_glob = args.input_file_pattern
    output_file_path = args.output_file
    dir, fname = os.path.split(output_file_path)
    img_dir = os.path.join(dir, "images_" + os.path.splitext(fname)[0])
    os.makedirs(img_dir, exist_ok=True)

    input_examples = get_examples(tfrecords_glob)

    inbreast_corrections = load_inbreast_corrections(args.inbreast_corrections)
    output_examples = convert(input_examples, img_dir, args.convert_generated, inbreast_corrections,
                              args.flip_label_gen)
    save_examples(output_examples, output_file_path)
