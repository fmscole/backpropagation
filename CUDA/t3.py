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
#     gdx=cuda.gridDim.x
#     gdy=cuda.gridDim.y
#     gdz=cuda.gridDim.z

#     bx=cuda.blockIdx.x
#     by=cuda.blockIdx.y
#     bz=cuda.blockIdx.z

    

#     bdx=cuda.blockDim.x
#     bdy=cuda.blockDim.y
#     bdz=cuda.blockDim.z

#     tx= cuda.threadIdx.x
#     ty= cuda.threadIdx.y
#     tz= cuda.threadIdx.z

#     h=gdx*bdx
#     w=gdy*bdy

    Batchs=A.shape[0]
#     Cout=C.shape[-1]
#     Cin=B.shape[-1]

    row, col,channel = cuda.grid(3)
#     if row < C.shape[0] and col < C.shape[1]:
#         tmp = 0.
        # for k in range(A.shape[1]):
        #     tmp += A[row, k] * B[k, col]
    for batch in range(Batchs):
        temp=0
        for cin in range(B.shape[2]):
                for kh in range(B.shape[1]):
                        for kw in range(B.shape[0]):
                                temp+=A[batch,row+kh,col+kw,cin]*B[batch,kh,kw,cin]
        C[batch,row,col,channel] =temp
@cuda.jit
def im2col(A,C,ksize=3,stride=1):
#     gdx=cuda.gridDim.x
    gdy=cuda.gridDim.y
#     gdz=cuda.gridDim.z

#     bdx=cuda.blockDim.x
    bdy=cuda.blockDim.y
#     bdz=cuda.blockDim.z

    
    Batchs=A.shape[0]
    row, col,channel = cuda.grid(3)
    for batch in range(Batchs):
        for cin in range(A.shape[-1]):
                for kh in range(ksize):
                        for kw in range(ksize):
                                C[batch,row*gdy*bdy+col,channel*ksize*ksize+kh*ksize+kw]=A[batch,row+kh,col+kw,cin]
         
def cuda_conv(x,K) :    
        # Host code
        # start=time.time()
        # Initialize the data arrays
        #BHWC
        B=x.shape[0]
        H=x.shape[1]
        W=x.shape[2]
        C=x.shape[3]
        k=K.shape[0]
        Cout=K.shape[-1]
        # x = numpy.full((B,H,W,C), 1, numpy.float) # matrix containing all 3's
        # kw = numpy.full((k, k,C,Cout), 1, numpy.float) # matrix containing all 4's

        # y=numpy.zeros(((B,H-k+1,W-k+1,Cout)))
        # Copy the arrays to the device
        A_global_mem = cuda.to_device(x)
        B_global_mem = cuda.to_device(K)
        # C_global_mem =cuda.to_device(y)
        # Allocate memory on the device for the result
        C_global_mem = cuda.device_array((B,H-k+1,W-k+1,Cout))

        # Configure the blocks: x*y*z<=1024
        threadsperblock = (H-k+1,W-k+1,1)
        # blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
        # blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
        blockspergrid =(1,1,Cout)

        # Start the kernel 
        matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

        # Copy the result back to the host
        C = C_global_mem.copy_to_host()
        # print(time.time()-start)
        # print(C.shape)
        # print(C[0,:,:,1])
        return C
def cuda_im2col(x,ksize=3,stride=1) :    
        B=x.shape[0]
        H=x.shape[1]
        W=x.shape[2]
        C=x.shape[3]
        A_global_mem = cuda.to_device(x)
        C_global_mem = cuda.device_array((B,(H-ksize+1)*(W-ksize+1),ksize*ksize*C))

        # Configure the blocks: x*y*z<=1024
        threadsperblock = (H-ksize+1,W-ksize+1,1)
        # blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
        # blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
        blockspergrid =(1,1,C)

        # Start the kernel 
        im2col[blockspergrid, threadsperblock](A_global_mem, C_global_mem,ksize,stride)

        # Copy the result back to the host
        C = C_global_mem.copy_to_host()
        # print(time.time()-start)
        # print(C.shape)
        # print(C[0,:,:,1])
        return C

# x = numpy.full((1,28,28,1), 1, numpy.float) # matrix containing all 3's
# kw = numpy.full((3, 3,1,2), 1, numpy.float) # matrix containing all 4's

# res= cuda_im2col(x=x)
# start=time.time()
# res= cuda_im2col(x=x)
# print(res[0,:,0])
# print(time.time()-start)