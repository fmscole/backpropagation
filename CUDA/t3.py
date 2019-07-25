from __future__ import division
from numba import cuda
import numpy
import math
import time
# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    tx= cuda.threadIdx.x
    ty= cuda.threadIdx.y
    tz= cuda.threadIdx.z

    bdx=cuda.blockDim.x
    bdy=cuda.blockDim.y
    bdz=cuda.blockDim.z

    bx=cuda.blockIdx.x
    by=cuda.blockIdx.y
    bz=cuda.blockIdx.z

    gdx=cuda.gridDim.x
    gdy=cuda.gridDim.y
    gdz=cuda.gridDim.z

    row, col,z = cuda.grid(3)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] =z
        
# Host code
start=time.time()
# Initialize the data arrays
A = numpy.full((12, 12), 3, numpy.float) # matrix containing all 3's
B = numpy.full((12, 11), 4, numpy.float) # matrix containing all 4's

# Copy the arrays to the device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

# Allocate memory on the device for the result
C_global_mem = cuda.device_array((12, 11))

# Configure the blocks: x*y*z<=1024
threadsperblock = (3,2,1)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (2, 2,10)

# Start the kernel 
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

# Copy the result back to the host
C = C_global_mem.copy_to_host()
# print(time.time()-start)
print(C)