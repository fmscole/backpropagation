import numpy as np
from functools import reduce
import math


class Dense(object):
    def __init__(self, input_num, output_num):
        self.input_len = input_num

        self.weights = np.random.normal(0.0, pow(self.input_len, -0.5), (self.input_len, output_num))
        self.bias = np.random.normal(0.0, pow(output_num, -0.5),output_num)
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.batchsize = x.shape[0]
        self.x = x.reshape(-1, self.input_len)
        output = np.dot(self.x, self.weights)+self.bias
        self.input_shape=x.shape
        return output

    def backward(self, eta):
        self.w_gradient=np.dot(self.x.T, eta)
        self.b_gradient=np.sum(eta,axis=0)
        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def gradient(self, alpha, weight_decay=0.0004):
        alpha=alpha /self.batchsize
        self.weights -= alpha* self.w_gradient
        self.bias -= alpha * self.b_gradient

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    fc = Dense(img.shape, 2)
    out = fc.forward(img)

    fc.gradient(np.array([[1, -2],[3,4]]))

    print (fc.w_gradient)
    print (fc.b_gradient)

    fc.backward()
    print (fc.weights)
