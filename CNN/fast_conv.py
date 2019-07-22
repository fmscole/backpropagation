import numpy as np
from functools import reduce
import math


class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        weights_scale = math.sqrt(
            reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (ksize,ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(
            self.output_channels) / weights_scale

        if method == 'VALID':
            self.eta = np.zeros((shape[0], int((shape[2] - ksize + 1) / self.stride), int((shape[2] - ksize + 1) / self.stride),
                                 self.output_channels))

        if method == 'SAME':
            self.eta = np.zeros(
                (shape[0], shape[1]/self.stride, shape[2]/self.stride, self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (shape[1] - ksize) % stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - ksize) % stride != 0:
            print('input tensor height can\'t fit stride')

    def forward(self, x):
        self.batchsize = x.shape[0]
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                'constant', constant_values=0)
        self.col_image=self.split_by_strides(x)
        conv_out=np.tensordot(self.col_image,self.weights, axes=([3,4,5],[0,1,2]))
        return conv_out

    def backward(self, eta):
        self.eta = eta
        col_image=self.col_image.transpose(3,4,5,0,1,2)
        self.w_gradient=np.tensordot(col_image,self.eta,axes=([3,4,5],[0,1,2]))

        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                'constant', constant_values=0)

        pad_eta=self.split_by_strides(pad_eta)
        weights=self.weights.transpose(0,1,3,2)
        next_eta=np.tensordot(pad_eta,weights, axes=([3,4,5],[0,1,2]))
        return next_eta

    def gradient(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha/self.batchsize * self.w_gradient
        self.bias -= alpha/self.batchsize * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def split_by_strides(self, x):
        # 将数据按卷积步长划分为与卷积核相同大小的子集,当不能被步长整除时，不会发生越界，但是会有一部分信息数据不会被使用
        N, H, W, C = x.shape
        oh = (H - self.ksize) // self.stride + 1
        ow = (W - self.ksize) // self.stride + 1
        shape = (N, oh, ow, self.ksize, self.ksize, C)
        strides = (x.strides[0], x.strides[1] * self.stride, x.strides[2] * self.stride, *x.strides[1:])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)





if __name__ == "__main__":
    # img = np.random.standard_normal((20, 32, 32, 3))
    # import time
    # start=time.time()
    # # img = np.ones((900, 900, 3))
    # # im2col(img, 3, 1)
    # # img = np.ones((20, 900, 900, 3))
    # # img *= 2
    # conv = Conv2D(img.shape, 12, 3, 1)
    # next = conv.forward(img)
    # next1 = next.copy() + 1
    # conv.gradient(next1-next)
    # # print(conv.w_gradient)
    # # print(conv.b_gradient)
    # # conv.backward()
    # print(time.time()-start)

    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread('15.png')
    img=np.array([img])
    
    print(img.shape)
    conv = Conv2D(img.shape, 3, 3, 1)
    next = conv.forward(img)
    # img2 = cv2.imread('test.jpg')
    print(next[0].shape)
    
    # plt.imshow(img[0])
    plt.imshow(next[0])
    
    # print(img[1, :, :, 1])
    # print(img1[1, :, :, 1])
    # print(img2[1, :, :, 1])
    # print(map(lambda x:int(x),img1[0]))
    # print(img1[0].shape)
    # plt.imshow(img1[0]/256)
    plt.show()

