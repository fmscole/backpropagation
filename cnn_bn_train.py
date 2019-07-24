import numpy as np
# from CNN.base_conv import Conv2D
from CNN.fast_conv import Conv2D
from CNN.fc import FullyConnect
# from CNN.pooling import MaxPooling as Pooling
from CNN.pooling import MeanPooling as Pooling
from CNN.softmax import Softmax
from CNN.relu import Relu
from CNN.my_read_Data import my_data_set
from CNN.batch_normal import BatchNormal as BN
from CNN.dropout import Dropout
from Net import Net
import time



data = my_data_set(kind='train')
test_data = my_data_set(kind='test')

batch_size = 167

# seq=[
#     Conv2D([batch_size,28, 28,1], 8, 5, 1),
#     BN(),
#     Relu(),
#     Pooling(),
#     Conv2D([batch_size,24, 24,8], 16, 5, 1),
#     BN(),
#     Relu(),
#     Dropout(p=0.5),
#     FullyConnect(1024, 200),
#     Relu(),
#     FullyConnect(200, 10),
#     Softmax()
# ]
# epochs=5

# seq=[
#     Conv2D([batch_size,28, 28,1], 8, 5, 1),
#     BN(),
#     Relu(),
#     Pooling(),
#     Conv2D([batch_size,24, 24,8], 16, 5, 1),
#     BN(),
#     Relu(),
#     Dropout(p=0.5),
#     Pooling(),
#     FullyConnect(256, 200),
#     # FullyConnect(1024, 200),
#     Relu(),
#     FullyConnect(200, 10),
#     Softmax()
# ]
# epochs=5

seq=[
    FullyConnect(784, 400),
    Relu(),
    FullyConnect(400, 10),
    Softmax()
]
epochs=50

net=Net(seq)
def  test():
    train_acc=0
    total=0
    batch_size=10000
    for i in range(10000//batch_size):
        imgs, labs = test_data.next_batch(batch_size)
        sf=net.forward(imgs,training=False)
        # train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf[j]) == np.argmax(labs[j]):
                train_acc += 1
            total+=1
        # print ("%d /%d   train_acc: %.4f  " % (i,10000//batch_size//10,train_acc / total))
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
        net.Gradient(alpha=learning_rate, weight_decay=0.001)

        #统计
        for j in range(batch_size):
            if np.argmax(sf_out[j]) == np.argmax(labs[j]):
                train_acc += 1

        mod = 100
        if i % mod == 0:
            print ("epoch=%d  batchs=%d   train_acc: %.4f  " % (epoch,i, train_acc / (mod * batch_size)))
            train_acc = 0

            print("----------------------------------------------------------------------------------------------------")
            print("epoch=",epoch,"     time(sec) is ",time.time()-start)
            start=time.time()
            test()
            print("----------------------------------------------------------------------------------------------------")
