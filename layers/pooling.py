import numpy as np
import matplotlib.pyplot as plt
import cv2
class MaxPooling(object):
    def __init__(self, size=2, **kwargs):
        '''
        size: Pooling的窗口大小，因为在使用中窗口大小与步长基本一致，所以简化为一个参数
        '''
        self.size = size
    def OutShape(self,shape):
        return [shape[0],shape[1]//self.size,shape[2]//self.size,shape[3]]

    def forward(self, x):
        # 首先将输入按照窗口大小划分为若干个子集
        #这个reshape方式非常精妙，把一个维度拆分为两个维度，并没有用滑动窗口的方式

        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])
        out = out.max(axis=(2, 4))
        

        #上面的两行代码等价于滑动窗口方式:        
        # N, H, W, C = x.shape
        # oh = (H - self.size) // self.size + 1
        # ow = (W - self.size) // self.size + 1
        # reshape = (N, oh, ow, self.size, self.size, C)
        # strides = (x.strides[0], x.strides[1] * self.size, x.strides[2] * self.size, *x.strides[1:])
        # out = np.lib.stride_tricks.as_strided(x,shape=reshape,strides=strides)
        # out = out.max(axis=(3, 4))

        # 记录每个窗口中不是最大值的位置
        self.mask = out.repeat(self.size, axis=1).repeat(self.size, axis=2) != x
        return out

    def backward(self, eta):
        # 将上一层传入的梯度进行复制，使其shape扩充到forward中输入的大小
        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
        # 将不是最大值的位置的梯度置为0
        eta[self.mask] = 0
        return eta
# 平均池化，用的很少，参考Maxpooling
class MeanPooling(object):
    def __init__(self,size=2, **kwargs):
        self.size = size
        
    def OutShape(self,shape):
        return [shape[0],shape[1]//self.size,shape[2]//self.size,shape[3]]

    def forward(self, x):
        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])
        return out.mean(axis=(2, 4))

    def backward(self, eta):
        return (eta / self.size**2).repeat(self.size, axis=1).repeat(self.size, axis=2)


if __name__ == "__main__":
    # img = cv2.imread('15.png')
    # # img=img[:400,:600]
    # img2=img
    # # img2 = cv2.imread('test.jpg')
    # print(img.shape)
    # img = np.array([img, img2]).reshape(
    #     [2, img.shape[0], img.shape[1], img.shape[2]])
    # print(img.shape) 
    # print(img[0].shape)    
    # # plt.imshow(img[0])
    # pool = MaxPooling(size=2)
    # img1 = pool.forward(img)
    # img2 = pool.backward(img1)
    # # print(img[1, :, :, 1])
    # # print(img1[1, :, :, 1])
    # # print(img2[1, :, :, 1])
    # # print(map(lambda x:int(x),img1[0]))
    # print(img1[0].shape)
    # plt.imshow(img1[0])
    # plt.show()
    #-----------------------------------------------------------------------------------------------------
    x=np.array(range(4*4)).reshape(1,4,4,1)
    print(x[0,:,:,0])
    pool=MaxPooling()
    y=pool.forward(x)
    print(y[0,:,:,0])
    z=pool.backward(y)
    print(z[0,:,:,0])
    #------------------------------------------------------------------------------------------------------
    # size=2
    # stride=1
    # x=np.array(range(4*4*3)).reshape(4,4,3)
    # print(x[:,:,0])

    # H, W,C= x.shape
    # oh = (H - size)  + 1
    # ow = (W - size) + 1
    # reshape = (oh, ow, size, size,C)
    # strides = (x.strides[0]*stride, x.strides[1] * stride, x.strides[0] * stride, x.strides[1],x.strides[-1])
    # out = np.lib.stride_tricks.as_strided(x,shape=reshape,strides=strides)
    # print(out[1,1,:,:,0])
    #------------------------------------------------------------------------------------
    # size=4
    # x=np.array(range(2*12*12*3)).reshape(2,12,12,3)
    # print(x[0,:,:,0])

    # N, H, W, C = x.shape
    # oh = (H - size) // size + 1
    # ow = (W - size) // size + 1
    # reshape = (N, oh, ow, size, size, C)
    # strides = (x.strides[0], x.strides[1] * size, x.strides[2] * size, *x.strides[1:])
    # out = np.lib.stride_tricks.as_strided(x,shape=reshape,strides=strides)

    # print(out[0,0,0,:,:,0])

    

