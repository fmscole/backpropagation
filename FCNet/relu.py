import numpy as np

class Relu(object):
    def __init__(self, shape=None):
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
    def __init__(self, shape=None):
        self.output_shape = shape

    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def gradient(self, eta):
        return eta*self.out*(1-self.out)

class Relu_Sigmoid(object):
    def __init__(self, shape=None):
        self.output_shape = shape
    def forward(self,x):
        self.x = x
        self.out = 1 / (1 + np.exp(-x))

        return np.maximum(x, 0)+self.out

    def gradient(self,eta):
        d1=eta * self.out * (1 - self.out)
        self.eta =np.copy(eta)
        self.eta[self.x < 0] = 0

class Softmax(object):
    def __init__(self, shape=None):
        self.output_shape = shape
    def forward(self, x):
        self.out =softmax(x)
        return  self.out
    def gradient(self, eta):
        # print ("out",self.out.shape)
        # print ("eta", eta.shape)
        dout=np.diag(self.out)-np.dot(self.out,self.out.T)
        # print("dout",dout.shape)
        return np.dot(dout,eta)
import copy
def softmax(x):
    #减去最大值
    tx=copy.copy(x)
    for i in range(tx.shape[1]):
        t=tx[:,i]
        t-=np.max(t)
        t=np.exp(t)
        s=np.sum(t)
        tx[:, i] = t/s
    return tx

