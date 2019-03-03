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

import argparse
import imgaug
from imgaug import augmenters as iaa
import numpy as np
import os

from resources.data.create_rcnn_dataset import get_examples, get_image, save_examples
from resources.data.features import to_feature_dict, example_to_str, example_to_int, to_example

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str, default=None)
parser.add_argument('--out_file', type=str, default=None)
args = parser.parse_args()

width = 204
height = 256

SEED = 42

k = 1

imgaug.seed(SEED)
aug = iaa.Sequential([
        iaa.Affine(rotate=(-4, 4)),
        iaa.Affine(scale={
                "x": (0.98, 1.13),
                "y": (0.98, 1.13)
        }),
])
aug_img = iaa.ContrastNormalization((0.08, 1.2), per_channel=False)


def augment_image(image, mask):
    image = aug_img.augment_image(image)

    aug_det = aug.to_deterministic()
    image = aug_det.augment_image(image)
    mask = aug_det.augment_image(mask)
    return image, mask


def augment_example(example):
    image = get_image(example, 'image')
    img_path = example_to_str(example, 'path')
    mask = get_image(example, 'mask')
    aug_image, aug_mask = augment_image(image, mask)
    lbl = example_to_int(example, 'label')
    # Save it all into Example proto
    assert image.shape[0] == height
    assert image.shape[1] == width
    features = to_feature_dict(
            img_path=img_path,
            img=image,
            mask=mask,
            width=np.int64(image.shape[1]),
            height=np.int64(image.shape[0]),
            label=np.int64(lbl))
    features.update(
            to_feature_dict(
                    img_path=img_path,
                    img=aug_image,
                    mask=aug_mask,
                    width=np.int64(image.shape[1]),
                    height=np.int64(image.shape[0]),
                    label=np.int64(lbl),
                    suffix='_gen'))
    return to_example(features)


def augment(in_file, out_file):

    def _augment():
        for example in get_examples(in_file):
            yield example  # Orig data.
            for _ in range(k):  # K-augmented data.
                yield augment_example(example)

    save_examples(_augment(), out_file)


os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
augment(args.in_file, args.out_file)
print('Done!')
