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

import functools
import imgaug
from imgaug import augmenters as iaa
from multiprocessing import Pool as ThreadPool
import numpy as np
import os
import plistlib
from skimage import filters
from skimage.draw import polygon
import tensorflow as tf

from resources.data.features import to_example, to_feature_dict
import resources.image_utils as imutils
import resources.synthetic_data as synth

COMPRESSION_TYPE = tf.python_io.TFRecordCompressionType.GZIP
OPTIONS = tf.python_io.TFRecordOptions(compression_type=COMPRESSION_TYPE)
SEED = 42

imgaug.seed(SEED)
aug = iaa.Sequential([iaa.Affine(rotate=(-4, 4)), iaa.Affine(scale={"x": (0.98, 1.13), "y": (0.98, 1.13)})])
aug_img = iaa.ContrastNormalization((0.08, 1.2), per_channel=False)


def generate_otsu_mask(img, timeout=20):  # generate larger masks
    mask_it = synth.generate_synth(size=img.shape, mask_only=True)

    def gen_otsu(img):
        val = filters.threshold_otsu(img)
        mask = img >= val
        return mask

    def gen_mask():
        return next(mask_it)

    otsu = gen_otsu(img)  # breast is true
    mask = gen_mask()  # cancer is non-0
    mask_and = np.zeros_like(otsu)
    for i in range(timeout):
        mask_bool = mask > 0
        np.logical_and(otsu, mask_bool, out=mask_and)
        assert mask_and.shape == mask_bool.shape
        if np.array_equal(mask_and, mask_bool):  # mask is inside otsu breast
            return mask
        mask = gen_mask()
    print("Didn't generate good mask in timeout ({}) steps, returning last, might be wrong.".format(timeout))
    return mask


def transform_single(img, img_meta, aug=None, aug_img=None, size=(256, 256), mask=False):
    # TODO: Possibly preprocess CBIS data by cutting 10px from all sides
    img = imutils.standardize(img, img_meta, mask=mask)
    img = imutils.downsample(img, size=size)

    if aug:
        img = imutils.normalize_gaussian(img)
        img = imutils.normalize(img, new_min=0, new_max=255)
        img = aug.augment_image(img)
        if aug_img:
            img = aug_img.augment_image(img)

    img = imutils.normalize_gaussian(img)
    img = imutils.normalize(img, new_min=0, new_max=255)
    return img


def transform_img(img, mask, img_meta, augment=False, size=(256, 256)):
    aug_det = None
    aug_image = None
    if augment:
        aug_det = aug.to_deterministic()
        aug_image = aug_img

    img = transform_single(img, img_meta, aug=aug_det, aug_img=aug_image, size=size, mask=False)
    mask = transform_single(mask, img_meta, aug=aug_det, aug_img=None, size=size, mask=True)
    return img, mask


def f(inp, fname_base, run_id, augment, label, size):
    thread_id, lst = inp
    label = np.int64(label)
    fname = "{}.r_{}.t_{}.tfrecord".format(fname_base, run_id, thread_id)
    imgaug.seed(SEED * run_id)
    with tf.python_io.TFRecordWriter(fname, options=OPTIONS) as writer:
        for img_meta in lst:
            try:
                img = imutils.load_image(img_meta.image_path)
            except ValueError as e:
                print("Failed to load image, skipping", img_meta.image_path)
                print(e)
                continue
            except AttributeError as e:
                print("Failed to load image, skipping", img_meta.image_path)
                print(e)
                continue
            try:
                mask = load_masks(img_meta.mask_path, img_meta.dataset, img)
            except ValueError as e:
                print("Failed to load mask, skipping", img_meta.mask_path)
                print(e)
                continue
            except AttributeError as e:
                print("Failed to load mask, skipping", img_meta.mask_path)
                print(e)
                continue
            img, mask = transform_img(img, mask, img_meta, augment=augment, size=size)
            height, width = img.shape
            example = to_example(to_feature_dict(img_meta.image_path, img, mask, width, height, label))
            writer.write(example.SerializeToString())


def transform_sequential(lst, folder, run_id, augment, label, size, threads=1, dataset=None):
    lst = (0, lst)
    print("Transforming (run_id={})".format(run_id))
    dataset = dataset if dataset is not None else 'all'
    fname_base = os.path.join(folder, "label_{}.augment_{}.dataset_{}".format(label, augment, dataset))
    f(lst, fname_base=fname_base, run_id=run_id, augment=augment, label=label, size=size)
    print("Transformed (run_id={})".format(run_id))


def transform_parallel(lst, folder, run_id, augment, label, size, threads=8, dataset=None):
    batch_size = len(lst) // threads + 1
    lst = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    lst = list(enumerate(lst))
    print("Transforming (run_id={})".format(run_id))
    pool = ThreadPool(threads)
    dataset = dataset if dataset is not None else 'all'
    fname_base = os.path.join(folder, "label_{}.augment_{}.dataset_{}".format(label, augment, dataset))
    print("Name:", fname_base)
    f_partial = functools.partial(f, fname_base=fname_base, run_id=run_id, augment=augment, label=label, size=size)
    pool.map(f_partial, lst)
    print("Transformed (run_id={})".format(run_id))


# df should be a list of image_meta
def transform(df, transformed_folder, cancer, augment_epochs=1, size=(256, 256), threads=1, dataset=None):
    os.makedirs(transformed_folder, exist_ok=True)
    transform_fn = transform_sequential if threads == 1 else transform_parallel

    for run_id in range(augment_epochs):
        transform_fn(
                df,
                transformed_folder,
                run_id,
                run_id != 0,
                label=int(cancer),
                size=size,
                threads=threads,
                dataset=dataset)


def merge_shards(in_files, fname):
    with tf.python_io.TFRecordWriter(fname, options=OPTIONS) as writer:
        for in_file in in_files:
            for record in tf.python_io.tf_record_iterator(in_file, options=OPTIONS):
                writer.write(record)


def merge_shards_in_folder(transformed_folder, transformed_out, size=(256, 256)):
    healthy = []
    cancer = []
    for file in os.listdir(transformed_folder):
        path = os.path.join(transformed_folder, file)
        if not file.startswith('label') or not file.endswith('tfrecord'):
            continue
        if file.startswith('label_1'):
            cancer.append(path)
        elif file.startswith('label_0'):
            healthy.append(path)
        else:
            raise ValueError('Invalid file: ' + str(path))

    merge_shards(healthy, os.path.join(transformed_out, 'healthy.tfrecord'))
    merge_shards(cancer, os.path.join(transformed_out, 'cancer.tfrecord'))


def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset

    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]

    return: numpy array where positions in the roi are assigned a value of 1.

    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x

    mask_shape = np.transpose(imshape)
    mask = np.zeros(mask_shape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [load_point(point) for point in points]
            if len(points) <= 2:
                for point in points:
                    mask[int(point[0]), int(point[1])] = 1
            else:
                x, y = zip(*points)
                x, y = np.array(x), np.array(y)
                poly_x, poly_y = polygon(x, y, shape=mask_shape)
                mask[poly_x, poly_y] = 1
    return mask


def load_bcdr_mask(lw_x_points_str, lw_y_points_str, imshape=(4084, 3328)):
    """
    This function loads a ROI region as a binary numpy array for BCDR_DXX
    dataset

    @lw_x_points_str : Iterable of string fields with x points (lw_x_points)
    @lw_y_points_str : Iterable of string fields with y points (lw_x_points)
    @imshape : The shape of the image as an array e.g. [4084, 3328]

    return: numpy array where positions in the roi are assigned a value of 1.

    """
    x_points = np.array([float(num) for num in lw_x_points_str.strip().split(' ')])
    y_points = np.array([float(num) for num in lw_y_points_str.strip().split(' ')])
    poly_x, poly_y = polygon(y_points, x_points, shape=imshape)
    mask = np.zeros((imshape))
    mask[poly_x, poly_y] = 1
    return mask


def load_masks(paths, dataset, img):
    if dataset == 'cbis':
        paths = list(paths)
        for path in paths:
            mask = imutils.load_image(path)
            if mask.shape == img.shape:
                return mask
        raise ValueError("CBIS: Couldn't load mask of same dimension as image, skipping.")
    elif dataset == 'inbreast':
        assert not isinstance(paths, list)
        if not os.path.isfile(paths):
            print("INBREAST: Couldn't load nonexistant mask, generating mask " + str(paths))
            return generate_otsu_mask(img)
        mask = load_inbreast_mask(paths, imshape=img.shape)
        assert mask.shape == img.shape
        return mask
    elif dataset == 'bcdr':
        if paths is None:
            print("BCDR: Couldn't load mask, generating mask.")
            return generate_otsu_mask(img)
        else:
            assert len(paths) == 2
            mask = load_bcdr_mask(paths[0], paths[1], imshape=img.shape)
            assert mask.shape == img.shape
            return mask
    elif dataset == 'zrh':
        # ZRH doesn't have masks.
        return generate_otsu_mask(img)
    else:
        raise ValueError('Unknown dataset ' + str(dataset))
