## sript to test pytorch speed up
## on gpu vs cpu

import torch
import time
import os

# Check PyTorch version and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set PyTorch to use all available CPU cores
num_threads = os.cpu_count()
torch.set_num_threads(num_threads)
print(f"Using {num_threads} CPU threads")

# Create a random tensor
a = torch.randn(33000, 33000)

# Function to measure time for matrix multiplication
def measure_time(device):
    a_device = a.to(device)
    start = time.time()
    c = torch.matmul(a_device, a_device)
    if device == 'cuda':
        torch.cuda.synchronize()  # Ensure all operations are finished for GPU
    end = time.time()
    print(f"Time taken for matrix multiplication on {device}: {end - start:.4f} seconds")
    return end - start

# Measure time on CPU
cpu_time = measure_time('cpu')

# Measure time on GPU (if available)
gpu_time = None
if torch.cuda.is_available():
    gpu_time = measure_time('cuda')

# Print comparison
if gpu_time:
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("No GPU detected.")
