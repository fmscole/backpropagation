import numpy as np
from base_conv import Conv2D
from fc import FullyConnect
from pooling import MaxPooling, AvgPooling
from softmax import Softmax
from relu import Relu
from my_read_Data import my_data_set

import time



data=my_data_set( kind='train')

batch_size = 17

conv1 = Conv2D([batch_size, 28, 28, 1], 12, 5, 1)
relu1 = Relu(conv1.output_shape)
pool1 = AvgPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 24, 3, 1)
relu2 = Relu(conv2.output_shape)
pool2 = AvgPooling(relu2.output_shape)
fc = FullyConnect(pool2.output_shape, 10)
fc1 = FullyConnect(pool2.output_shape, 200)
relu3 = Relu(fc1.output_shape)
fc2 = FullyConnect(relu3.output_shape, 10)
sf = Softmax(fc2.output_shape)


for epoch in range(1):
    
    learning_rate =1

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    # imgs,labs=data.next_batch(batch_size)
    for i in range(100000):
        imgs,labs=data.next_batch(batch_size)
        conv1_out = relu1.forward(conv1.forward(imgs))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        # fc_out1 = relu3.forward(fc1.forward(pool2_out))
        # fc_out = fc2.forward(fc_out1)
        sf.cal_loss(fc_out, labs)
        # train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == np.argmax(labs[j]):
                train_acc += 1

        sf.gradient()
        conv1.gradient(relu1.gradient(pool1.gradient(
            conv2.gradient(relu2.gradient(pool2.gradient(
                fc.gradient(sf.eta)))))))
                # fc1.gradient(relu3.gradient(fc2.gradient(sf.eta)))))))))
        # fc1.gradient(relu1.gradient(fc2.gradient(sf.eta)))

        
        # fc2.backward(alpha=learning_rate, weight_decay=0.0004)
        # fc1.backward(alpha=learning_rate, weight_decay=0.0004)
        fc.backward(alpha=learning_rate, weight_decay=0.0004)
        conv2.backward(alpha=learning_rate, weight_decay=0.0004)
        conv1.backward(alpha=learning_rate, weight_decay=0.0004)

        mod=10
        if i % mod == 0:
            print ("i=%d   train_acc: %.4f  " % (i, train_acc /(mod* batch_size)))
            train_acc=0


