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

import resources.data.loader as loader
import resources.data.transformer as transformer
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--cancer', type=str2bool, default=True)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=208)
parser.add_argument('--augment_epochs', type=int, default=1)
parser.add_argument('--merge', type=str2bool, default=False)
parser.add_argument('--in_folder', type=str, default='/scratch_net/biwidl104/oskopek')
parser.add_argument('--out_folder', type=str, default=None)
args = parser.parse_args()

print('PyDataset:', args.dataset)
print('PyCancer:', args.cancer)
print('PyMerge:', args.merge)

breast_prefix = os.path.abspath(args.in_folder)
loader.init(breast_prefix)

size = (args.height, args.width)

os.makedirs(args.out_folder, exist_ok=True)
if args.merge:
    transformer.merge_shards_in_folder(args.in_folder, args.out_folder)
else:

    healthy, cancer = loader.load_all_datasets(dataset=args.dataset)

    if args.cancer:
        df = cancer
    else:
        df = healthy
    transformer.transform(
            df,
            args.out_folder,
            args.cancer,
            threads=args.threads,
            dataset=args.dataset,
            size=size,
            augment_epochs=args.augment_epochs)

print('Done!')
