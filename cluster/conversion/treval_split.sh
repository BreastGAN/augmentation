#!/bin/bash

user=$USER
size="256x204"

dataserver="biwidl100"

src_dir="${1:-/scratch_net/$dataserver/$user/transformed/traineval_$size}"
dst_dir="${2:-/scratch_net/$dataserver/$user/transformed/small_all_${size}_final}"

healthy="$src_dir/healthy.tfrecord"
cancer="$src_dir/cancer.tfrecord"

test_src_dir="${1:-/scratch_net/$dataserver/$user/transformed/test_$size}"
cp "$test_src_dir/healthy.tfrecord" "$dst_dir/healthy.test.tfrecord"
cp "$test_src_dir/cancer.tfrecord" "$dst_dir/cancer.test.tfrecord"

#
# Script to send job to BIWI clusters using qsub.
# Usage: qsub run_on_host.sh model models.breast_cycle_gan_graph flags/cyclegan.json

# Adjust line '-l hostname=xxxxx' before running.
# The script also requires changing the paths of the CUDA and python environments
# and the code to the local equivalents of your machines.
# Author: Christian F. Baumgartner (c.f.baumgartner@gmail.com)

## SET THE FOLLOWING VARIABLES ACCORDING TO YOUR SYSTEM ##
BASE_HOME=/scratch_net/$dataserver/$USER
PROJECT_HOME=$BASE_HOME/mammography
VIRTUAL_ENV_PATH=$PROJECT_HOME/venv-cpu

## SGE Variables:
#
## otherwise the default shell would be used
#$ -S /bin/bash
#
## <= 2h is short queue, <= 24h is middle queue, <= 120h is long queue
#$ -l h_rt=02:00:00

## the maximum memory usage of this job, (below 4G does not make much sense)
#$ -l h_vmem=10G  # Less RAM is required for evaluating than for training

# Host and gpu settings
##$ -l gpu
##$ -l hostname=biwirender08   ## <-------------- Comment in or out to force a specific machine
#$ -l hostname=bender58

## stderr and stdout are merged together to stdout
#$ -j y
#
# logging directory. preferably on your scratch
#$ -o /scratch_net/biwidl100/oskopek/logs/  ## <---------------- CHANGE TO MATCH YOUR SYSTEM
#
## send mail on job's end and abort
##$ -m a ## Mails are disabled

## LOCAL PATHS
# I think .bashrc is not executed on the remote host if you use qsub, so you need to set all the paths
# and environment variables before exectuting the python code.

# for pyenv
export PATH="/home/oskopek/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

## EXECUTION OF PYTHON CODE:
echo "Hostname is: `hostname`"
echo "Run model:"
cd $PROJECT_HOME
exec $VIRTUAL_ENV_PATH/bin/python -m notebooks.treval_split --healthy "$healthy" --cancer "$cancer" --out_dir="$dst_dir" --train_percent 0.99

