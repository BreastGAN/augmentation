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
from PIL import Image
import pydicom
import skimage.transform
import skimage.io


def show_img(img):
    # Lazy import
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2)
    ax.imshow(img)
    ax2.hist(np.ravel(img))
    plt.show()


def to_numpy(img):
    width, height = img.size
    return np.reshape(img.getdata(), (height, width)).astype(np.float64)


def load_dicom(path):
    dcm = pydicom.dcmread(path)
    img = np.copy(dcm.pixel_array)
    return np.asarray(img).astype(np.float64)


def load_with_pil(path):
    return to_numpy(Image.open(path))


def load_image(path):
    if path.endswith('.dcm'):
        return load_dicom(path)
    elif path.endswith('.tif') or path.endswith('.png'):
        return load_with_pil(path)
    else:
        raise ValueError('Unknown file format for file {}'.format(path))


def save_image(image, file):
    image = np.reshape(image, (256, 256))
    skimage.io.imsave(file, image)


def standardize(img, img_meta, mask=False):
    # TODO: check if CBIS is rotated properly and remove
    def is_breast_left(img):
        width2 = img.shape[1] // 2
        left_mean = np.mean(img[:, :width2])
        right_mean = np.mean(img[:, width2:])
        return left_mean > right_mean

    if img_meta.dataset == 'cbis':
        print('Fixing cbis laterality')
        img_meta.laterality = 'L' if is_breast_left(img) else 'R'
    if img_meta.dataset == 'zrh':
        print('Fixing zrh laterality')
        img_meta.laterality = 'L' if is_breast_left(img) else 'R'

    if img_meta.laterality == 'R':  # horizontal flip for right breasts
        img = np.flip(img, axis=1)
    # TODO(#6) what about MLO view?
    if img_meta.dataset == 'cbis' and not mask:  # untag
        img[:1000, -1000:] = 0
    elif img_meta.dataset == 'zrh' and not mask:  # untag
        img[:75, -100:] = 0
    return img


def normalize_gaussian(img, mean=None, std=None):
    """
    Normalizes an image by the mean and dividing by std (Gaussian normalization).

    If mean or std is None, uses np.mean or np.std respectively.
    """
    if mean is None:
        mean = np.mean(img)
    if std is None:
        std = np.std(img)
    img = img - mean
    if std != 0:
        img = img / std  # Needed for comparable histograms!
    return img


def normalize(img, new_min=0, new_max=255):
    """
    Normalizes an image by linear transformation into the interval [new_min, new_max].
    """
    old_min = np.min(img)
    old_max = np.max(img)
    if old_min == old_max:
        return img - old_min  # return 0s
    img = (img - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    return img


def downsample(img, size=(256, 128)):
    scale = max(size) / max(img.shape)
    img = skimage.transform.rescale(img, scale, mode='constant', multichannel=False, anti_aliasing=True)
    img_new = np.zeros(size)
    min_h = min(size[0], img.shape[0])
    min_w = min(size[1], img.shape[1])
    img_new[:min_h, :min_w] = img[:min_h, :min_w]  # Fill in to full size x size, or crop from right.
    # TODO(#6) Is this a good idea?
    assert img_new.shape == size
    return img_new
