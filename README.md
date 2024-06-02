# gpu_testing
Testing performance of new GPU

Repo for setting up my new laptop with an NVIDIA GPU and testing how good it is.


## Install tools

In powershell, First install wsl and set to wsl2

```
wsl --install
wsl --set-default-version 2
```

In wsl, install python
```
sudo apt update
sudo apt install python3 python3-pip
```

Install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Install latest cuda
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

Set environment variables
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Install cuDNN
```
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz
tar -xvf cudnn-*.tar.xz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Check correct cuda version is installed
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Apr_17_19:19:55_PDT_2024
Cuda compilation tools, release 12.5, V12.5.40
Build cuda_12.5.r12.5/compiler.34177558_0
```


## Install ML libraries

Create three separate environments for testing:
1. tensorflow-env: just for tensorflow (no pytorch)
2. pytorch-env: just pytorch (no tensorflow)
3. tf-torch-env: both tensorflow and pytorch


Tensorflow:
```
conda create --name tensorflow-env python=3.12
conda update conda
conda install -c conda-forge tensorflow
```

Pytorch:
```
conda create --name pytorch-env python=3.12
conda update conda
conda install -c pytorch pytorch torchvision torchaudio cudatoolkit=11.8 # other cudatoolkit versions are possible but some did not work ie >=12
```

Both: # note this setup blocked tensorflow from using the GPU for me...
```
conda create --name tf-torch-env python=3.12
conda update conda
conda install -c conda-forge tensorflow
conda install -c conda-forge pytorch torchvision torchaudio cudatoolkit=11.8
```


## Test GPU acceleration

Some basic scripts are provided to test matrix multiplication on a random 10000, 10000 tensor using either CPU or GPU.

[Tensorflow alone:](tf_test.py)
```
conda deactivate
conda activate tensorflow-env
python tf_test.py
# Time taken for matrix multiplication on /CPU:0 : 5.391279220581055 seconds
# Time taken for matrix multiplication on /GPU:0 : 0.059388160705566406 seconds
# CPU time: 5.3989 seconds
# GPU time: 0.0612 seconds
# Speedup: 88.20x
```

[Pytorch alone:](torch_test.py)
```
conda deactivate
conda activate pytorch-env
python torch_test.py
# PyTorch version: 2.3.0
# CUDA available: True
# Using 20 CPU threads
# Time taken for matrix multiplication on cpu: 4.9638 seconds
# Time taken for matrix multiplication on cuda: 0.3238 seconds
# CPU time: 4.9638 seconds
# GPU time: 0.3238 seconds
# Speedup: 15.33x
```


[Side-by-side (both):](tf-torch-test.py)
In this env tensorflow is not finding the GPU for some reason...
```
conda deactivate
conda activate tf-torch-env
python tf-torch-test.py
# Using 20 CPU threads for PyTorch
# TensorFlow version: 2.16.1
# PyTorch version: 2.3.0
# CUDA available (TensorFlow): []
# CUDA available (PyTorch): True
# Time taken for matrix multiplication on /CPU:0 : 4.963411331176758 seconds
# Time taken for matrix multiplication on cpu: 5.1325 seconds
# Time taken for matrix multiplication on cuda: 0.3272 seconds
# CPU time (TensorFlow): 4.9699 seconds
# CPU time (PyTorch): 5.1325 seconds
# No GPU detected.
```