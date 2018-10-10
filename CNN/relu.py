import numpy as np

class Relu(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta

class sigmoid(object):
    def __init__(self, shape):
        self.output_shape = shape

    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def gradient(self, eta):
        return eta*self.out*(1-self.out)

class Relu_Sigmoid(object):
    def __init__(self, shape):
        self.output_shape = shape
    def forward(self,x):
        self.x = x
        self.out = 1 / (1 + np.exp(-x))

        return np.maximum(x, 0)+self.out

    def gradient(self,eta):
        d1=eta * self.out * (1 - self.out)
        self.eta =np.copy(eta)
        self.eta[self.x < 0] = 0

        return d1+self.eta