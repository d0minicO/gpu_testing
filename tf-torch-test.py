## script to test the performance of pytorch
## and tensorflow using either cpu or gpu

## Dominic Owens
## 2024-06-02

import torch
import tensorflow as tf
import time
import os

# Set the number of threads for PyTorch
num_threads = os.cpu_count()
torch.set_num_threads(num_threads)
print(f"Using {num_threads} CPU threads for PyTorch")

# TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available (TensorFlow): {tf.config.list_physical_devices('GPU')}")
print(f"CUDA available (PyTorch): {torch.cuda.is_available()}")

# Create random tensors
a_tf = tf.random.normal([10000, 10000])
a_torch = torch.randn(10000, 10000)

# Function to measure time for matrix multiplication in TensorFlow
def measure_time_tf(device):
    with tf.device(device):
        start = time.time()
        c = tf.matmul(a_tf, a_tf)
        tf.print("Time taken for matrix multiplication on", device, ":", time.time() - start, "seconds")
        return time.time() - start

# Function to measure time for matrix multiplication in PyTorch
def measure_time_torch(device):
    a_device = a_torch.to(device)
    start = time.time()
    c = torch.matmul(a_device, a_device)
    if device == 'cuda':
        torch.cuda.synchronize()  # Ensure all operations are finished for GPU
    end = time.time()
    print(f"Time taken for matrix multiplication on {device}: {end - start:.4f} seconds")
    return end - start

# Measure time on CPU
cpu_time_tf = measure_time_tf('/CPU:0')
cpu_time_torch = measure_time_torch('cpu')

# Measure time on GPU (if available)
gpu_time_tf = None
gpu_time_torch = None
if tf.config.list_physical_devices('GPU'):
    gpu_time_tf = measure_time_tf('/GPU:0')

if torch.cuda.is_available():
    gpu_time_torch = measure_time_torch('cuda')

# Print comparison
print(f"CPU time (TensorFlow): {cpu_time_tf:.4f} seconds")
print(f"CPU time (PyTorch): {cpu_time_torch:.4f} seconds")
if gpu_time_tf and gpu_time_torch:
    print(f"GPU time (TensorFlow): {gpu_time_tf:.4f} seconds")
    print(f"GPU time (PyTorch): {gpu_time_torch:.4f} seconds")
    print(f"Speedup (TensorFlow CPU -> GPU): {cpu_time_tf / gpu_time_tf:.2f}x")
    print(f"Speedup (PyTorch CPU -> GPU): {cpu_time_torch / gpu_time_torch:.2f}x")
else:
    print("No GPU detected.")

