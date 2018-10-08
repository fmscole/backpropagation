#python3
#源码来源， https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork，有部分修改
import numpy
import copy
from batch_normal import BatchNormal as bn
def sigmoid(x):
    return 1/(1+numpy.exp(-x))

def softmax(x):
    #减去最大值
    t=copy.copy(x)
    t-=numpy.max(t)
    t=numpy.exp(t)
    s=numpy.sum(t)
    t = t/s
    return t

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes #=784
        self.hnodes = hiddennodes #=200
        self.onodes = outputnodes #=10
        self.m=0 #计算训练过程中的准确率，否则训练过程太乏味了
        self.n=0 #计算训练进度，看不到进度的训练太烦人
        
        #最重要的两个权重矩阵
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        #偏置
        self.bh = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes,1 ))
        self.bo = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes,1 ))
        # 学习率=0.1
        self.lr = learningrate
        # 激活函数sigmoid
        self.activation_function =sigmoid
        self.bn=bn()
    def train(self, inputs, targets):
        batchs=inputs.shape[1]

        #向前计算        
        hidden_inputs = numpy.dot(self.wih, inputs)+self.bh
        hidden_inputs = self.bn.forward(hidden_inputs, axis=-1)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)+self.bo
        final_outputs =self.activation_function(final_inputs)
        # final_outputs =numpy.array([softmax(final_inputs[:,i]) for i in range(batchs)]).T
        
        #计数训练次数
        self.n = self.n + 1
        # 处理批量准确率计算
        for i in range(batchs):
            t = (final_outputs[:,i].argmax()==targets[:,i].argmax() )
            if t: self.m = self.m + 1
        
        #每cm批次算一次准确率        
        cm=100  
        if self.n % cm == 0:
            cc = self.m / (batchs*cm)
            print(self.n, cc)
            self.m = 0

        output_errors = targets - final_outputs
        # 在who更新之前，先把梯度存起来，更新完了就传不过去了
        hidden_errors = numpy.dot(self.who.T, output_errors) 

        # 为了适应批量训练，这个地方要做小幅修改，要除以训练数量batchs，更新为平均值
        self.who += self.lr/batchs * numpy.dot(output_errors ,
                 numpy.transpose(hidden_outputs))
        hidden_errors=hidden_errors* hidden_outputs * (1.0 - hidden_outputs)
        hidden_errors=self.bn.backward(hidden_errors)
        # 同样，这里也要除以训练数量batchs
        self.wih += self.lr/batchs * numpy.dot(hidden_errors, numpy.transpose(inputs))
        
        self.bh+= self.lr/batchs* numpy.sum(hidden_errors,axis=1,keepdims=True)
        self.bo+= self.lr/batchs* numpy.sum(output_errors,axis=1,keepdims=True) 
        
    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)+self.bh
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)+self.bo
        final_outputs =self.activation_function(final_inputs)
        
        return final_outputs