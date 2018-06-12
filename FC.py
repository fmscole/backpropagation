#python3
#源码来源， https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork，有部分修改
import numpy

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

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

        # 学习率=0.1
        self.lr = learningrate
        # 激活函数sigmoid
        self.activation_function =sigmoid

    def train(self, inputs, targets):
        batchs=inputs.shape[1]
        #向前计算        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs =self.activation_function(final_inputs)

        #计算训练过程中的准确率
        self.n = self.n + 1
        # 处理批量训练情况
        for i in range(batchs):
            t = (targets[:,i].argmax() == final_outputs[:,i].argmax())
            if t: self.m = self.m + 1
        #每cm批次算一次准确率        
        cm=100  
        if self.n % cm == 0:
            cc = self.m / (targets.shape[1]*cm)
            print(self.n, cc)
            self.m = 0

        output_errors = targets - final_outputs
        # 在who更新之前，先把梯度存起来，更新完了就传不过去了
        hidden_errors = numpy.dot(self.who.T, output_errors) 

        # 为了适应批量训练，这个地方要做小幅修改，要除以训练数量batchs，更新为平均值
        self.who += self.lr/batchs * numpy.dot(output_errors ,
                 numpy.transpose(hidden_outputs))
        # 同样，这里也要除以训练数量batchs
        self.wih += self.lr/batchs * numpy.dot((hidden_errors 
            * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
             
    
    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs