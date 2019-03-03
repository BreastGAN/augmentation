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
from PIL import Image
import tensorflow as tf


def get_image(example, suffix=''):
    w = example.features.feature['width'].int64_list.value[0]
    h = example.features.feature['height'].int64_list.value[0]
    img_string = example.features.feature['image' + suffix].bytes_list.value[0]
    img = np.frombuffer(img_string, dtype=np.float32)
    img = img.reshape(h, w)
    return img


def get_images(tfrecords_glob,
               options=tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)):
    for file in tf.gfile.Glob(tfrecords_glob):
        for record in tf.python_io.tf_record_iterator(file, options=options):
            example = tf.train.Example()
            example.ParseFromString(record)
            yield get_image(example), get_image(example, suffix='_gen')


def to_png(matrix, path):
    im = Image.fromarray(matrix)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(path)


chkpt = 0
# infdir = '/scratch_net/biwidl104/oskopek/inferred/MaskFalse_BcdrInbreastFilterTrain'
infdir = './data_out/MaskFalse_BcdrInbreastFilterTrain'
infdir += '_NoAugment_ICNR_nnUPSAMPLE_Lam0.0_SpectralNorm_steps_{}_inference_eval'.format(chkpt)
for cancer in ["cancer.eval", "healthy.eval"]:
    src_glob = os.path.join(infdir, "{}_gen.tfrecord".format(cancer))
    target_dir = os.path.join(infdir + "_png_{}".format(cancer))
    print(src_glob, target_dir)
    tf.gfile.MakeDirs(target_dir)
    for i, (image, gen) in enumerate(get_images(src_glob)):
        to_png(image, os.path.join(target_dir, "{:05}_orig.png".format(i)))
        to_png(gen, os.path.join(target_dir, "{:05}_gen.png".format(i)))
