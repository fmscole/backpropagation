import numpy as np
# from layers.base_conv import Conv2D
from layers.fast_conv import Conv2D
from layers.Dense import Dense
from layers.pooling import MaxPooling ,MeanPooling
from layers.softmax import Softmax
from layers.relu import Relu,Sigmoid
from layers.my_read_Data import my_data_set
from layers.batch_normal import BatchNormal as BN
from layers.dropout import Dropout
from Net import Net
import time

class res_block(object):
    def __init__(self, ksize=3, stride=1):
        self.stride = stride
        self.ksize = ksize
    def OutShape(self,shape):
        self.conv1=Conv2D(output_channels=8,ksize=self.ksize,stride=self.stride,method="SAME")
        self.relu1=Relu()
        self.conv2=Conv2D(output_channels=shape[-1],ksize=self.ksize,stride=self.stride,method="SAME")
        self.bn=BN()
        self.relu2=Relu()

        out_shape=shape
        out_shape=self.conv1.OutShape(out_shape)
        out_shape=self.conv2.OutShape(out_shape)

        return out_shape

    def forward(self, x):
        out=x
        out=self.conv1.forward(out)
        out=self.relu1.forward(out)
        out=self.conv2.forward(out)
        out=self.bn.forward(out)
        out=self.relu2.forward(out)
        return x+out
    def backward(self, eta):
        out=eta
        out=self.relu2.backward(out)
        out=self.bn.backward(out)
        out=self.conv2.backward(out)
        out=self.relu1.backward(out)
        out=self.conv1.backward(out)
        return eta+out
    def gradient(self, alpha=0.00001, weight_decay=0.0004):
        self.conv1.gradient(alpha=alpha,weight_decay=weight_decay)
        self.conv2.gradient(alpha=alpha,weight_decay=weight_decay)



