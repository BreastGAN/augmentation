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

import nbformat
from yapf.yapflib.yapf_api import FormatCode

style_file = '.style.yapf'


def format_nb(notebook_filename, dry_run=False):
    print('Formatting {}...'.format(notebook_filename), end='')
    with open(notebook_filename, 'r') as f:
        notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)
    nbformat.validate(notebook)

    changed = False
    if notebook['nbformat'] != 4 or notebook['nbformat_minor'] != 0:
        notebook['nbformat_minor'] = 0
        changed = True

    for token in ['colab', 'language_info']:
        if token in notebook.metadata:
            del notebook.metadata[token]
            changed = True

    for cell in notebook.cells:
        if cell['metadata'] != {}:
            cell['metadata'] = {}
            changed = True

    for cell in notebook.cells:
        if cell['cell_type'] != 'code':
            continue

        src = cell['source']
        lines = src.split('\n')
        if len(lines) <= 0 or '# noqa' in lines[0]:
            continue

        formatted_src, did_change = FormatCode(src, style_config=style_file)
        if did_change:
            cell['source'] = formatted_src
            changed = True

    if changed:
        if not dry_run:
            with open(notebook_filename, 'w') as f:
                nbformat.write(notebook, f, version=4)
        print(' (reformatted)')
    else:
        print()


def main(notebook_filenames, dry_run=False):
    for fn in notebook_filenames:
        format_nb(fn, dry_run=dry_run)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file_to_format', nargs='+', help='The jupyter notebook file to format in place.')
    parser.add_argument(
            '--dry_run', action='store_true', help='Whether to just print if a notebook would be reformatted.')
    args = parser.parse_args()

    main(args.file_to_format, dry_run=args.dry_run)
