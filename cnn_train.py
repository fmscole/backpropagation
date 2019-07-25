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
from layers.res_block import res_block
from Net import Net
import time


data = my_data_set(kind='train')
test_data = my_data_set(kind='test')

#超参数
batch_size = 17
learning_rate = 0.01
epochs=500
#定义网络结构
#不同的参数和结构会有不同的准确率，由于每次初始化也会影响到准确率，
#下面这个网络在epochs=5时准确率在98.6%到98.75%之间
#下面这个网络在epochs=87时准确率99%
#每个epoch耗时85秒
# seq=[
#     Conv2D(output_channels=8,ksize=5,stride=1),
#     BN(),
#     Relu(),
#     MaxPooling(),
#     Conv2D(output_channels=16, ksize=5, stride=1),
#     BN(),
#     Relu(),
#     # Sigmoid(),
#     Dropout(p=0.2),
#     Dense(output_num=200),
#     Relu(),
#     Dense( output_num=10),
#     # Sigmoid()
#     Softmax()
# ]

#残差网络
#每个epoch耗时97秒
seq=[
    Conv2D(output_channels=8,ksize=5,stride=1),
    MaxPooling(),
    res_block(),
    res_block(),
    Dropout(p=0.2),
    Dense( output_num=10),
    Softmax()
]
epochs=5

#没有BN层，训练不够稳定，开始若干个batch准确率都没有提升，
#学习率learning_rate对结果影响比较大，甚至不收敛
# epochs=5
# seq=[
#     Conv2D(output_channels=8,ksize=3,stride=1),
#     # BN(),
#     Relu(),
#     MaxPooling(),
#     Conv2D(output_channels=16, ksize=3, stride=1),
#     # BN(),
#     Sigmoid(),
#     Dropout(p=0.5),
#     Dense(output_num=200),
#     Relu(),
#     Dense( output_num=10),
#     Softmax()
# ]


#最简单的神经网络，准确率92%
# epochs=50
# seq=[
#     Dense(output_num=10),
#     Sigmoid()
# ]


#input_shape必须是BHWC的顺序，如果不是，需要reshape和tanspose成NHWC顺序
net=Net(seq=seq,input_shape=[batch_size,28, 28,1])

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
    

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    total=0
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
            total+=1

        mod = 100
        if i % mod == 0:
            print ("epoch=%d  batchs=%d   train_acc: %.4f  " % (epoch,i, train_acc / total))
            train_acc = 0
            total=0

    print("----------------------------------------------------------------------------------------------------")
    print("epoch=",epoch," batchs=%d      time is:  %5.5f (sec)"%(i,time.time()-start))
    start=time.time()
    test()
    print("----------------------------------------------------------------------------------------------------")
