import numpy  
from  FC  import neuralNetwork
import pickle as pk 
from my_read_Data import my_data_set

#测试函数
def test(n):
    test_data=my_data_set(kind='test')
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
    print ("performance = ", scorecard_array.sum() / scorecard_array.size,scorecard_array.sum(),scorecard_array.size)


#构建神经网络
input_nodes = 784
hidden_nodes = 400
output_nodes = 10

learning_rate = 0.1

n2 = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

fl = open('temp100w.pkl', 'rb')
n2.bo=pk.load(fl)
n2.bo=numpy.reshape(n2.bo,[-1,1])
n2.bh=pk.load(fl).reshape((-1,1))
n2.who=pk.load(fl)
n2.wih=pk.load(fl)
fl.close()

test(n2)
# print(n2.bh)