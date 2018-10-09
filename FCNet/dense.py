import numpy as np


class Dense():
    def __init__(self, inodes, hnodes):
        self.inodes = inodes
        self.hnodes = hnodes
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.bh = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, 1))
    def forward(self, inputs):
        self.inputs=inputs
        self.batch_size=inputs.shape[1]
        hidden_inputs = np.dot(self.wih, inputs) #+ self.bh
        return hidden_inputs
    def backward(self,eta,lr=0.1):
        dout = np.dot(self.wih.T, eta)
        db=lr/self.batch_size*np.sum(eta,axis=1,keepdims=True)
        dw= lr / self.batch_size * np.dot(eta, np.transpose(self.inputs))
        self.bh+=db
        self.wih+=dw
        return dout