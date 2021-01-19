#NOTE: Initialize conda (pyenv or virtualenv) before running the script!!!
#conda create --name tf --yes python=3.8
#conda activate tf

echo 'Adding NVIDIA package repositories'
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

echo 'Installing NVIDIA driver'
sudo apt-get install --no-install-recommends nvidia-driver-450

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

echo 'Installing development and runtime libraries (~4GB)'
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

echo 'Installing TensorRT. Requires that libcudnn8 is installed above.'
sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0


echo 'Installing tensorflow=2.4 and gretel-synthetics'
pip3 install --upgrade pip
pip3 install tensorflow==2.4 gretel-synthetics

echo 'Check if tensorflow with gpu support is installed'
python -c "import tensorflow as tf; tf.test.gpu_device_name()"
