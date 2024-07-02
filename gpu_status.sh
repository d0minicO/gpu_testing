#!/bin/bash
# Script to check and print status of CUDA and GPU

echo "Checking NVIDIA GPU status..."
nvidia-smi

echo "Checking CUDA version..."
nvcc --version

echo "Checking GPU details..."
lspci | grep -i nvidia

echo "Checking CUDA libraries..."
ls /usr/local/cuda/lib64
