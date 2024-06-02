# Script to demonstrate speedup of using vectorised dot product
# instead of for loop when performing dot product on linear regression
# weights and input feature values for prediction

import numpy as np
import time

# initialise random vectors of this length
n = int(1e8)
# as a dummy for linear regression weights
w = np.random.rand(n)
# and for input feature values
x = np.random.randint(0,101, size=n)

# function for manually performing dot product using a for loop
def measure_time_forloop(w,x):
    start = time.time()
    f = 0
    for i in range(n):
        f = f + (w[i]*x[i])

    return time.time() - start

# function for properly performing dot product using vectorisation
def measure_time_vectorised(w,x):
    start = time.time()
    f = np.dot(w,x)

    return time.time() - start


for_time = measure_time_forloop(w,x)
vec_time = measure_time_vectorised(w,x)

print(f"for loop time: {for_time:.4f} seconds")
print(f"vectorised time: {vec_time:.4f} seconds")
