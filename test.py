#python3
import numpy
from FC import neuralNetwork
from my_read_Data import my_data_set

#测试函数
def test():
    test_data=my_data_set()
    test_data.load_mnist( kind='test')
    images_test,labels_test=test_data.next_batch(10000)

    #矩阵运算放在循环外面比在里面速度快很多
    outputs = n.query(images_test)

    scorecard = []
    for j in range(outputs.shape[1]):  
        if (numpy.argmax(outputs[:,j]) == numpy.argmax(labels_test[:,j])):
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)


#构建神经网络
input_nodes = 784
hidden_nodes = 400
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

#读取数据
data=my_data_set()
data.load_mnist( kind='train')

epochs = 10
for e in range(epochs):
    for i  in range(60000):
        #小批量训练
        imgs,labs=data.next_batch(37)
        n.train(imgs, labs)
    test()