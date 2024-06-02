## script for testing tensorflow speed up
## on gpu vs cpu

import tensorflow as tf
import time

# Create a random tensor
a = tf.random.normal([33000, 33000])

# Function to measure time for matrix multiplication
def measure_time(device):
    with tf.device(device):
        start = time.time()
        c = tf.matmul(a, a)
        tf.print("Time taken for matrix multiplication on", device, ":", time.time() - start, "seconds")
        return time.time() - start

# Measure time on CPU
cpu_time = measure_time('/CPU:0')

# Measure time on GPU (if available)
gpu_time = None
if tf.config.experimental.list_physical_devices('GPU'):
    gpu_time = measure_time('/GPU:0')

# Print comparison
if gpu_time:
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("No GPU detected.")
