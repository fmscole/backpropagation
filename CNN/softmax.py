import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
class Softmax(object):
    def forward(self, x):
        '''
        x.shape = (N, C)
        接收批量的输入，每个输入是一维向量
        计算公式为：
        a_{ij}=\frac{e^{x_{ij}}}{\sum_{j}^{C} e^{x_{ij}}}
        '''
        v = np.exp(x - x.max(axis=-1, keepdims=True))    
        return v / v.sum(axis=-1, keepdims=True)
    
    def backward(self, y):
        # 一般Softmax的反向传播和CrossEntropyLoss的放在一起
        pass
    # def cal_loss(self, y,t):
        