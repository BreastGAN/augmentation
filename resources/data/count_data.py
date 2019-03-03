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
import argparse
from collections import Counter
from resources.data.features import example_to_str

OPTIONS = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
parser = argparse.ArgumentParser()
parser.add_argument("--input_file_pattern", help="Global pattern for input tfrecords.")
parser.add_argument("--count_generated", help="Whether to count generated images as well.", action='store_true')
args = parser.parse_args()
assert args.input_file_pattern

tfrecords_glob = args.input_file_pattern


def get_labels(path):
    print("Processing file {}".format(path))
    for record in tf.python_io.tf_record_iterator(path, options=OPTIONS):
        example = tf.train.Example()
        example.ParseFromString(record)
        if args.count_generated:
            if example_to_str(example, 'path').endswith("_gen"):
                yield "gen-{}".format(example.features.feature['label'].int64_list.value[0])
            else:
                yield str(example.features.feature['label'].int64_list.value[0])
        else:
            yield example.features.feature['label'].int64_list.value[0]


counter = 0
for file in tf.gfile.Glob(tfrecords_glob):
    c = Counter(get_labels(file))
    print("File {}: {}, total: {}".format(file, c, sum(c.values())))
    counter += sum(c.values())

print("Total number of examples: ", counter)
