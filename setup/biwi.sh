echo "# settings for BIWI cluster" >> ~/.bashrc
echo "source /home/sgeadmin/BIWICELL/common/settings.sh" >> ~/.bashrc

echo 'export LANG="en_US.UTF-8"' >> ~/.bashrc
echo 'export LC_ALL="$LANG"' >> ~/.bashrc

git clone https://github.com/pyenv/pyenv.git ~/.pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
pyenv install 3.5.5
pyenv global 3.5.5
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
mkdir /scratch_net/biwidl104/$USER
ln -s /scratch_net/biwidl104/$USER ~/scratch
#ln -s /usr/biwimaster01/data-biwi-01/$USER ~/data

export SGE_GPU=`grep -h $(whoami) /tmp/lock-gpu*/info.txt | sed  's/^[^0-9]*//;s/[^0-9].*$//'`
export CUDA_VISIBLE_DEVICES="$SGE_GPU"
echo 'export SGE_GPU=`grep -h $(whoami) /tmp/lock-gpu*/info.txt | sed  "s/^[^0-9]*//;s/[^0-9].*\$//"`' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES="$SGE_GPU"' >> ~/.bashrc

# TODO: Setup ssh keys!

source ~/.bashrc

cd ~/scratch
# TODO: Follow: https://docs.google.com/document/d/1UXhXkqn20v_jC3CzSvdgvgED2iuXIKsgcW16GtVYBu8/edit#
# But with CUDA 9.0
mkdir -p cuda
cd cuda
wget 'https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run' -O cuda9.run
sh cuda9.run
# Enter Toolkit Location
# [ default is /usr/local/cuda-9.0 ]: ~/scratch/cuda/cuda-9.0

exit

# manual steps:

# cuDNN
# Download Download cuDNN v7.0.5 for Linux (Mar 21, 2018), for CUDA 9.0 through the NVIDIA page
# scp it into scratch_net/biwidl104/$USER
# place it in scratch/cudnn, extract into cudnn/cuda using tar
# scp cudnn-9.0-linux-x64-v7.tgz brossa.ee.ethz.ch:/scratch_net/biwidl104/oskopek/cudnn/
# tar xfz cudnn-9.0-linux-x64-v7.tgz

echo 'export CUDA_HOME="`realpath $HOME/scratch/cuda/cuda-9.0`"' >> ~/.bashrc
echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc

export CUDA_HOME="`realpath $HOME/scratch/cuda/cuda-9.0`"
CUDNN_UNCOMPRESSED_FOLDER="`realpath $HOME/scratch/cudnn/cuda`"
cp -v $CUDNN_UNCOMPRESSED_FOLDER/include/*.*  $CUDA_HOME/include
cp -v $CUDNN_UNCOMPRESSED_FOLDER/lib64/*.*  $CUDA_HOME/lib64
chmod a+r $CUDA_HOME/include/cudnn.h $CUDA_HOME/lib64/libcudnn*

echo 'alias gup="$HOME/scratch/mammography/resources/biwi/update_repo.sh"' >> ~/.bashrc
echo 'alias gst="git status"' >> ~/.bashrc
echo 'alias g="git"' >> ~/.bashrc
echo 'alias train="qsub ~/scratch/mammography/resources/biwi/run_on_host.sh model"' >> ~/.bashrc

source ~/.bashrc
cd ~/scratch/
git clone git@gitlab.com:breast-cancer-eth/mammography.git
cd mammography
./setup/create_venv.sh

# Copy data from local PC: (make sure you're logged in on brossa when doing this
# scp data_in/*.zip brossa.ee.ethz.ch:/scratch_net/biwidl104/oskopek/mammography/data_in/ <<- Your username here

# Mount the scratch on your PC (for tensorboard)
# sshfs brossa.ee.ethz.ch:/scratch_net/biwidl104/oskopek biwi
# Umount: fusermount -u ~/biwi
