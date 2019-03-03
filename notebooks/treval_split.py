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

import tensorflow as tf

import os

import argparse
from sklearn.model_selection import train_test_split
from resources.data.transformer import OPTIONS

parser = argparse.ArgumentParser()
parser.add_argument('--healthy', type=str, default=None)
parser.add_argument('--cancer', type=str, default=None)
parser.add_argument('--out_dir', type=str, default=None)
parser.add_argument('--train_percent', type=float, default=0.85)
args = parser.parse_args()


def treval_split(healthy, cancer, train_percentage):
    healthy_labels = [0] * len(healthy)
    cancer_labels = [1] * len(cancer)
    lst = healthy + cancer
    labels = healthy_labels + cancer_labels
    train_lst, eval_lst, train_labels, eval_labels = train_test_split(
            lst, labels, stratify=labels, train_size=train_percentage, random_state=42)

    healthy_train, healthy_eval = [], []
    cancer_train, cancer_eval = [], []
    for ex, label in zip(train_lst, train_labels):
        if label == 1:
            cancer_train.append(ex)
        else:
            healthy_train.append(ex)
    for ex, label in zip(eval_lst, eval_labels):
        if label == 1:
            cancer_eval.append(ex)
        else:
            healthy_eval.append(ex)
    return healthy_train, cancer_train, healthy_eval, cancer_eval


def read_unparsed(records_file):
    res = []
    for record in tf.python_io.tf_record_iterator(records_file, options=OPTIONS):
        res.append(record)
    return res


def write_unparsed(records_lst, records_file):
    with tf.python_io.TFRecordWriter(records_file, options=OPTIONS) as writer:
        for record in records_lst:
            writer.write(record)


os.makedirs(args.out_dir, exist_ok=True)

healthy = read_unparsed(args.healthy)
cancer = read_unparsed(args.cancer)

healthy_train, cancer_train, healthy_eval, cancer_eval = treval_split(healthy, cancer, args.train_percent)

write_unparsed(healthy_train, os.path.join(args.out_dir, 'healthy.train.tfrecord'))
write_unparsed(cancer_train, os.path.join(args.out_dir, 'cancer.train.tfrecord'))
write_unparsed(healthy_eval, os.path.join(args.out_dir, 'healthy.eval.tfrecord'))
write_unparsed(cancer_eval, os.path.join(args.out_dir, 'cancer.eval.tfrecord'))

print('Done!')
