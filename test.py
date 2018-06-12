#python3
import numpy
from FC import neuralNetwork
from my_read_Data import my_data_set

#测试函数
def test():
    test_data=my_data_set()
    images_test,labels_test=test_data.load_mnist( kind='test')
    
    scorecard = []
    for j in range(10000):   
        outputs = n.query(images_test[j])
        label = numpy.argmax(outputs)
        correct_label = numpy.argmax(labels_test[j])
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)


#构建神经网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

#读取数据
data=my_data_set()
data.load_mnist( kind='train')

epochs = 1
for e in range(epochs):
    for i  in range(600):
        #小批量训练
        imgs,labs=data.next_batch(37)
        n.train(imgs, labs)
    test()