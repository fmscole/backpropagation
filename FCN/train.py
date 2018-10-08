#python3
import numpy
from FC import neuralNetwork
from my_read_Data import my_data_set
import pickle as pk 


#构建神经网络
input_nodes = 784
hidden_nodes = 400
output_nodes = 10
learning_rate = 0.1

if __name__ =='__main__':
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    #读取数据
    data=my_data_set( kind='train')
    
    epochs = 1
    for e in range(epochs):
        for i  in range(100000):
            #小批量训练
            imgs,labs=data.next_batch(17)
            n.train(imgs, labs)

    fs = open('temp20.pkl', 'wb')
    pk.dump(n.bo,fs)
    pk.dump(n.bh,fs)
    pk.dump(n.who,fs)
    pk.dump(n.wih,fs)
    fs.close()

    

    