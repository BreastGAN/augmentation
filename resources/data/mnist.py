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

from tensorflow.examples.tutorials import mnist
from resources.data.utils import shuffle


def print_format_info(no_gpu):
    device, data_format = ('/gpu:0', 'channels_first')
    if no_gpu:
        device, data_format = ('/cpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))


def read_data_sets(data_dir, one_hot=True):
    """Returns training and test tf.data.Dataset objects."""
    data = mnist.input_data.read_data_sets(data_dir, one_hot=one_hot)
    return (data.train, data.test)


# Read MNIST + batch and shuffle it.
def read_mnist(in_dir, no_gpu=False):
    print_format_info(no_gpu)
    train_ds, _ = read_data_sets(in_dir)
    return shuffle(train_ds.images, train_ds.labels)


# Read MNIST + batch and shuffle it. Output only numbers of one class.
def read_mnist_label(in_dir, label=2, no_gpu=False):

    def create_mask(dataset, label):
        mask = []
        for _, lab in zip(dataset.images, dataset.labels):
            if lab == label:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def read_data_sets_label(data_dir, label):
        """Returns training and test tf.data.Dataset objects."""
        train_data, test_data = read_data_sets(data_dir, one_hot=False)
        train_mask = create_mask(train_data, label)
        test_mask = create_mask(test_data, label)
        return (train_data.images[train_mask], test_data.images[test_mask])

    # Load the datasets
    print_format_info(no_gpu)
    train_ds, _ = read_data_sets_label(in_dir, label)
    return train_ds
