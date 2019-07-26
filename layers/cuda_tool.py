from __future__ import division
from numba import cuda
import numpy
import math
import time
# CUDA kernel
@cuda.jit
def conv(A, B, C):
    Batchs=A.shape[0]
    row, col,channel = cuda.grid(3)

    if row>=A.shape[1] or col>=A.shape[2] or channel>=C.shape[3]: return

    for batch in range(Batchs):
        temp=0
        for cin in range(B.shape[2]):
                for kh in range(B.shape[1]):
                        for kw in range(B.shape[0]):
                                temp+=A[batch,row+kh,col+kw,cin]*B[batch,kh,kw,cin]
        C[batch,row,col,channel] =temp

def cuda_conv(x,K) :    
        B=x.shape[0]
        H=x.shape[1]
        W=x.shape[2]
        k=K.shape[0]

        Cout=K.shape[-1]
        A_global_mem = cuda.to_device(x)
        B_global_mem = cuda.to_device(K)
        C_global_mem = cuda.device_array((B,H-k+1,W-k+1,Cout))
        # threadsperblock = (,,1)
        # blockspergrid =(1,1,)

        threadsperblock = (16,16,4)
        blockspergrid =(int(math.ceil((H-k+1) / threadsperblock[0])),
                        int(math.ceil((W-k+1) / threadsperblock[1])),
                        int(math.ceil(Cout / threadsperblock[2])))

        conv[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
        return C_global_mem.copy_to_host()
#-----------------------------------------------------------------------------------------
@cuda.jit
def batchdot(A, B, C):
    batch,row, col= cuda.grid(3)
    temp=0
    if batch>=A.shape[0] or row>=A.shape[1] or col>=B.shape[1]: return
    for i in range(A.shape[2]):
        temp+=A[batch,row,i]*B[i,col]
    C[batch,row,col]=temp

def cuda_dot(x,y) :    
        B=x.shape[0]
        H=x.shape[1]
        W=y.shape[1]

        A_global_mem = cuda.to_device(x)
        B_global_mem = cuda.to_device(y)
        C_global_mem = cuda.device_array((B,H,W))

        threadsperblock = (8,32,4)
        blockspergrid =(int(math.ceil(B / threadsperblock[0])),
                        int(math.ceil(H / threadsperblock[1])),
                        int(math.ceil(W / threadsperblock[2])))
        batchdot[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
        return C_global_mem.copy_to_host()   
# x=numpy.array(range(2*16*17)).reshape(2,16,17)
# y=numpy.array(range(17*5)).reshape(17,5)
# print(cuda_dot(x,y))
#-------------------------------------------------------------------------------------------
@cuda.jit
def im2col2(A,C,ksize,stride):
    Batchs=A.shape[0]
    row, col,channel = cuda.grid(3)
    dcol=A.shape[2]-ksize+1
    if row>= A.shape[1]-ksize+1 or col>=A.shape[2]-ksize+1 or channel>=A.shape[3]: return
    # C[0,row*gdy*bdy+col,channel*ksize*ksize+0*ksize+0]=0.1
    for batch in range(Batchs):
        for cin in range(A.shape[3]):
                for kh in range(ksize):
                        for kw in range(ksize):
                                C[batch,row*dcol+col,channel*ksize*ksize+kh*ksize+kw]=A[batch,row+kh,col+kw,cin]

def cuda_im2col2(x,ksize,stride=1) :    
        B=x.shape[0]
        H=x.shape[1]
        W=x.shape[2]
        C=x.shape[3]
        A_global_mem = cuda.to_device(x)
        C_global_mem = cuda.device_array((B,(H-ksize+1)*(W-ksize+1),ksize*ksize*C))

        
        threadsperblock = (16,16,4)
        blockspergrid =(int(math.ceil((H-ksize+1) / threadsperblock[0])),
                        int(math.ceil((W-ksize+1) / threadsperblock[1])),
                        int(math.ceil(C / threadsperblock[2])))

        im2col[blockspergrid, threadsperblock](A_global_mem, C_global_mem,ksize,stride)

        C = C_global_mem.copy_to_host()
        return C

# x=numpy.array(range(2*16*17*3)).reshape(2,16,17,3)
# y=numpy.array(range(17*5)).reshape(17,5)
# print(cuda_im2col(x))
# print(cuda_im2col(x))
#-------------------------------------------------------------------------------------------

@cuda.jit
def batchdot2(A, B, C):
    batch=cuda.blockIdx.x #B
    row=cuda.blockIdx.y #H
    col=cuda.blockIdx.z #W

    i= cuda.threadIdx.x #kh

    # batch,row, col= cuda.grid(3)
    # temp=0
    # if batch>=A.shape[0] or row>=A.shape[1] or col>=B.shape[1]: return
    # for i in range(A.shape[2]):
    #     temp+=A[batch,row,i]*B[i,col]
    C[batch,row,col]+=A[batch,row,i]*B[i,col]

def cuda_dot2(x,y) :    
        B=x.shape[0]
        H=x.shape[1]
        i=x.shape[2]
        W=y.shape[1]

        A_global_mem = cuda.to_device(x)
        B_global_mem = cuda.to_device(y)
        C_global_mem = cuda.device_array((B,H,W))

        threadsperblock = (i,1,1)
        blockspergrid =(B,H,W)
        batchdot2[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
        return C_global_mem.copy_to_host()   
# x=numpy.array(range(2*16*17)).reshape(2,16,17)
# y=numpy.array(range(17*5)).reshape(17,5)
# print(cuda_dot(x,y))
#-------------------------------------------------------------------------------------------

@cuda.jit
def im2col(A,C,ksize,stride):
    bx=cuda.blockIdx.x #B
    by=cuda.blockIdx.y #H
    bz=cuda.blockIdx.z #W

    tx= cuda.threadIdx.x #kh
    ty= cuda.threadIdx.y #kw
    tz= cuda.threadIdx.z #C

    dcol=A.shape[2]-ksize+1

    C[bx,by*dcol+bz,tz*ksize*ksize+tx*ksize+ty]=A[bx,by+tx,bz+ty,tz]

def cuda_im2col(x,ksize,stride=1) :    
        B=x.shape[0]
        H=x.shape[1]
        W=x.shape[2]
        C=x.shape[3]
        A_global_mem = cuda.to_device(x)
        C_global_mem = cuda.device_array((B,(H-ksize+1)*(W-ksize+1),ksize*ksize*C))

        
        threadsperblock = (ksize,ksize,C)
        blockspergrid =(B,H-ksize+1,W-ksize+1)

        im2col2[blockspergrid, threadsperblock](A_global_mem, C_global_mem,ksize,stride)

        C = C_global_mem.copy_to_host()
        return C

# x=numpy.array(range(2*16*17*3)).reshape(2,16,17,3)
# y=numpy.array(range(17*5)).reshape(17,5)
# print(cuda_im2col2(x,ksize=3))
# print(cuda_im2col(x))
#-------------------------------------------------------------------------------------------

