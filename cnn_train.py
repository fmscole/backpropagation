import numpy as np
# from layers.base_conv import Conv2D
from layers.fast_conv import Conv2D
from layers.Dense import Dense
from layers.pooling import MaxPooling as Pooling
# from layers.pooling import MeanPooling as Pooling
from layers.softmax import Softmax
from layers.relu import Relu
from layers.my_read_Data import my_data_set
from layers.batch_normal import BatchNormal as BN
from layers.dropout import Dropout
from Net import Net
import time



data = my_data_set(kind='train')
test_data = my_data_set(kind='test')

batch_size = 167
#加BN层
seq=[
    Conv2D([batch_size,28, 28,1], 8, 5, 1),
    BN(),
    Relu(),
    Pooling(),
    Conv2D([batch_size,12, 12,8], 16, 5, 1),
    BN(),
    Relu(),
    Dropout(p=0.5),
    Dense(1024, 200),
    Relu(),
    Dense(200, 10),
    Softmax()
]
epochs=5

#没有BN层
# seq=[
#     Conv2D([batch_size,28, 28,1], 8, 5, 1),
#     Relu(),
#     Pooling(),
#     Conv2D([batch_size,12, 12,8], 16, 5, 1),
#     Relu(),
#     Dropout(p=0.5),
#     Dense(1024, 200),
#     Relu(),
#     Dense(200, 10),
#     Softmax()
# ]
# epochs=5

#最简答的神经网络
# seq=[
#     Dense(784, 10),
#     Softmax()
# ]
# epochs=50

net=Net(seq)
def  test():
    train_acc=0
    total=0
    batch_size=10000
    for i in range(10000//batch_size):
        imgs, labs = test_data.next_batch(batch_size)
        sf=net.forward(imgs,training=False)
        for j in range(batch_size):
            if np.argmax(sf[j]) == np.argmax(labs[j]):
                train_acc += 1
            total+=1
    print ("Test_acc: %.4f  " % (train_acc / total))

for epoch in range(epochs):
    start=time.time()
    learning_rate = 0.05

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    # imgs,labs=data.next_batch(batch_size)
    for i in range(60000//batch_size):
        imgs, labs = data.next_batch(batch_size)
        imgs=imgs

        #训练
        sf_out=net.forward(imgs)
        net.backward(sf_out-labs)
        net.Gradient(alpha=learning_rate, weight_decay=0.01)

        #统计
        for j in range(batch_size):
            if np.argmax(sf_out[j]) == np.argmax(labs[j]):
                train_acc += 1

        mod = 100
        if i % mod == 0:
            print ("epoch=%d  batchs=%d   train_acc: %.4f  " % (epoch,i, train_acc / (mod * batch_size)))
            train_acc = 0

            print("----------------------------------------------------------------------------------------------------")
            print("epoch=",epoch," batchs=%d      time is:  %5.5f (sec)"%(i,time.time()-start))
            start=time.time()
            test()
            print("----------------------------------------------------------------------------------------------------")
