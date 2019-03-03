#!/bin/bash

set -e

if ./setup/check_venv.sh
then
    echo "Not in venv, please activate it first."
    exit 1
fi

nbstripout notebooks/*.ipynb
python resources/yapf_nbformat/yapf_nbformat.py notebooks/*.ipynb
yapf -r -i . -e venv-cpu/ -e resources/colab_utils/ -e venv/ -e data_in/ -e data_out/ -e '**/.ipynb_checkpoints/'
