import numpy as np
A=np.arange(16).reshape(4,4)
xstep=1;ystep=1; xsize=3; ysize=3
print(((A.shape[0] - xsize + 1) // xstep,(A.shape[1] - ysize + 1) //ystep,xsize, ysize))
window_view = np.lib.stride_tricks.as_strided(A, ((A.shape[0] - xsize + 1) // xstep,
                                                  (A.shape[1] - ysize + 1) //ystep, 
                                                  xsize, ysize),
                                              (A.strides[0] * xstep,
                                               A.strides[1] * ystep,
                                               A.strides[0], 
                                               A.strides[1]))
print((A.strides[0] * xstep,A.strides[1] * ystep,A.strides[0],A.strides[1]))
print(window_view.shape)
window_view=window_view.reshape(-1,9)
print(A)
print()
print(window_view)
