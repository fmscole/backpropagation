import numpy as np
from functools import reduce
import math
from layers.cuda_tool import cuda_im2col,cuda_dot2

class Conv2D(object):
    def __init__(self,output_channels, ksize=3, stride=1, method='VALID'):
        self.output_channels = output_channels
        self.stride = stride
        self.ksize = ksize
        self.method = method
    def OutShape(self,shape):
        self.input_shape=shape
        self.input_channels = shape[-1]
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal((self.ksize*self.ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        
        if (shape[1] - self.ksize) % self.stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - self.ksize) % self.stride != 0:
            print('input tensor height can\'t fit stride')

        if self.method == 'VALID':
            return [shape[0], 
                    (shape[1] - self.ksize + 1) // self.stride, 
                    (shape[1] - self.ksize + 1) // self.stride,
                    self.output_channels]
        # self.method == 'SAME':
        return [shape[0], 
                shape[1]// self.stride, 
                shape[2]// self.stride, 
                self.output_channels]

    def forward(self, x):
        shape=x.shape
        if self.method == 'VALID':
            self.eta = np.zeros((shape[0], int((shape[1] - self.ksize + 1) / self.stride), int((shape[1] - self.ksize + 1) / self.stride),
                                 self.output_channels))
        if self.method == 'SAME':
            self.eta = np.zeros((shape[0], shape[1]//self.stride, shape[2]//self.stride, self.output_channels))

        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                'constant', constant_values=0)

        col_weights = self.weights.reshape([-1, self.output_channels])
        self.col_image = []
        self.batchsize=x.shape[0]
        conv_out = np.zeros(self.eta.shape)
        
        self.col_image=cuda_im2col(x,self.ksize)
        # for i in range(self.batchsize):
        #     img_i = x[i]
        #     self.col_image_i = im2col(img_i, self.ksize, self.stride)
        #     self.col_image.append(self.col_image_i)
        # self.col_image = np.array(self.col_image)
        conv_out=np.dot(self.col_image, col_weights)
        # conv_out=cuda_dot2(self.col_image, col_weights)
        
        conv_out=np.reshape(conv_out, self.eta.shape)
        return conv_out

    def backward(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T,
                                      col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                'constant', constant_values=0)

        flip_weights=self.weights[::-1,...]
        flip_weights = flip_weights.swapaxes(1, 2)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])

        # col_pad_eta = np.array([im2col(pad_eta[i], self.ksize, self.stride) for i in range(self.batchsize)])
        

        col_pad_eta=cuda_im2col(pad_eta,self.ksize)
        next_eta = np.dot(col_pad_eta, col_flip_weights)

        # next_eta = np.dot(col_pad_eta, col_flip_weights)

        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def gradient(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        # self.weights *= (1 - weight_decay)
        # self.bias *= (1 - weight_decay)
        self.weights -= alpha/self.batchsize * self.w_gradient
        self.bias -= alpha/self.batchsize * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[0] - ksize + 1, stride):
        for j in range(0, image.shape[1] - ksize + 1, stride):
            col = image[i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


if __name__ == "__main__":
    # # img = np.random.standard_normal((2, 32, 32, 3))
    # import time
    # start=time.time()
    # img = np.ones((200, 32, 32, 3))
    # img *= 2
    # conv = Conv2D(img.shape, 12, 3, 1)
    # next = conv.forward(img)
    # next1 = next.copy() + 1
    # conv.gradient(next1-next)
    # # print(conv.w_gradient)
    # # print(conv.b_gradient)
    # conv.backward()
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

