import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

import numpy as np
from CNN.base_conv import Conv2D
from CNN.fc import FullyConnect
from CNN.pooling import MaxPooling, AvgPooling
from CNN.softmax import Softmax
from CNN.relu import Relu
from CNN.my_read_Data import my_data_set
from CNN.batch_normal import BatchNormal as BN
import time

data = my_data_set(kind='train')

batch_size = 17

conv1 = Conv2D([batch_size, 28, 28, 1], 12, 5, 1)
bn1=BN()
relu1 = Relu(conv1.output_shape)
pool1 = AvgPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 24, 3, 1)
relu2 = Relu(conv2.output_shape)
pool2 = AvgPooling(relu2.output_shape)
fc1 = FullyConnect(pool2.output_shape, 200)
relu3 = Relu(fc1.output_shape)
fc2 = FullyConnect(relu3.output_shape, 10)
sf = Softmax(fc2.output_shape)

for epoch in range(1):

    learning_rate = 1

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    # imgs,labs=data.next_batch(batch_size)
    for i in range(100000):
        imgs, labs = data.next_batch(batch_size)
        conv1_out=conv1.forward(imgs)
        relu1_out = relu1.forward(conv1_out)
        pool1_out = pool1.forward(relu1_out)
        conv2_out=conv2.forward(pool1_out)
        relu2_out = relu2.forward(conv2_out)
        pool2_out = pool2.forward(relu2_out)
        fc1_out1=fc1.forward(pool2_out)
        relu3_out1 = relu3.forward(fc1_out1)
        fc2_out = fc2.forward(relu3_out1)
        sf.cal_loss(fc2_out, labs)
        # train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == np.argmax(labs[j]):
                train_acc += 1

        sf.gradient()
        gfc2=fc2.gradient(sf.eta)
        grelu3=relu3.gradient(gfc2)
        gfc1=fc1.gradient(grelu3)
        gpool2=pool2.gradient(gfc1)
        grelu2=relu2.gradient(gpool2)
        gconv2=conv2.gradient(grelu2)
        gpool1=pool1.gradient(gconv2)
        grelu1=relu1.gradient(gpool1)
        conv1.gradient(grelu1)
        # fc1.gradient(relu1.gradient(fc2.gradient(sf.eta)))

        fc2.backward(alpha=learning_rate, weight_decay=0.0004)
        fc1.backward(alpha=learning_rate, weight_decay=0.0004)
        # fc.backward(alpha=learning_rate, weight_decay=0.0004)
        conv2.backward(alpha=learning_rate, weight_decay=0.0004)
        conv1.backward(alpha=learning_rate, weight_decay=0.0004)

        mod = 10
        if i % mod == 0:
            print ("i=%d   train_acc: %.4f  " % (i, train_acc / (mod * batch_size)))
            train_acc = 0


