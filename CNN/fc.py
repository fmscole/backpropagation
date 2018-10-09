import numpy as np
from functools import reduce
import math


class FullyConnect(object):
    def __init__(self, shape, output_num):
        self.input_shape = shape
        self.batchsize = shape[0]

        input_len = reduce(lambda x, y: x * y, shape[1:])

        # self.weights = np.random.standard_normal((input_len, output_num))/100
        # self.bias = np.random.standard_normal(output_num)/100

        self.weights = np.random.normal(0.0, pow(input_len, -0.5), (input_len, output_num))
        self.bias = np.random.normal(0.0, pow(output_num, -0.5),output_num)

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, eta):
        # for i in range(eta.shape[0]):
        #     col_x = self.x[i][:, np.newaxis]
        #     eta_i = eta[i][:, np.newaxis].T
        #     self.w_gradient += np.dot(col_x, eta_i)
        #     self.b_gradient += eta_i.reshape(self.bias.shape)
        self.w_gradient=np.dot(self.x.T, eta)
        self.b_gradient=np.sum(eta,axis=0)
        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha, weight_decay=0.0004):
        # weight_decay = L2 regularization
        # self.weights *= (1 - weight_decay)
        # self.bias *= (1 - weight_decay)
        alpha=np.min( [alpha /self.batchsize,0.01])
        # alpha=np.max( alpha /self.batchsize,0.001)
        self.weights -= alpha* self.w_gradient
        self.bias -= alpha * self.b_gradient
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    fc = FullyConnect(img.shape, 2)
    out = fc.forward(img)

    fc.gradient(np.array([[1, -2],[3,4]]))

    print (fc.w_gradient)
    print (fc.b_gradient)

    fc.backward()
    print (fc.weights)