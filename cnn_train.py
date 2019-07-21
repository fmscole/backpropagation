import numpy as np
# from CNN.base_conv import Conv2D
from CNN.fast_conv import Conv2D
from CNN.fc import FullyConnect
from CNN.pooling import MaxPooling, AvgPooling
from CNN.softmax import Softmax
from CNN.relu import Relu,sigmoid
from CNN.my_read_Data import my_data_set
from CNN.batch_normal import BatchNormal as BN
import time

data = my_data_set(kind='train')
test_data = my_data_set(kind='test')

batch_size = 17

conv1 = Conv2D([batch_size, 28, 28,1], 12, 5, 1)
bn1=BN()
relu1 = Relu(conv1.output_shape)
pool1 = AvgPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 24, 3, 1)
bn2=BN()
relu2 = Relu(conv2.output_shape)
pool2 = AvgPooling(relu2.output_shape)
fc1 = FullyConnect(pool2.output_shape, 200)
relu3 = Relu(fc1.output_shape)
fc2 = FullyConnect(relu3.output_shape, 10)
sf = Softmax(fc2.output_shape)

def  test():
    train_acc=0
    total=0
    for i in range(50):
        imgs, labs = test_data.next_batch(batch_size)
        forward(imgs, labs,training=False)
        # train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == np.argmax(labs[j]):
                train_acc += 1
            total+=1
        print ("%d /%d   train_acc: %.4f  " % (i,10000//batch_size//10,train_acc / total))
    print (" train_acc: %.4f  " % (train_acc / total))


def forward(imgs, labs,training=True):
    conv1_out = conv1.forward(imgs)
    bn1_out=bn1.forward(conv1_out, axis=3,training=training)
    relu1_out = relu1.forward(bn1_out)
    pool1_out = pool1.forward(relu1_out)

    conv2_out = conv2.forward(pool1_out)
    bn2_out = bn2.forward(conv2_out, axis=3,training=training)
    relu2_out = relu2.forward(bn2_out)
    pool2_out = pool2.forward(relu2_out)

    fc1_out1 = fc1.forward(pool2_out)
    relu3_out1 = relu3.forward(fc1_out1)
    fc2_out = fc2.forward(relu3_out1)
    sf.cal_loss(fc2_out, labs)

start=time.time()
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
        forward(imgs, labs)

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == np.argmax(labs[j]):
                train_acc += 1

        sf.gradient()
        gfc2=fc2.gradient(sf.eta)
        grelu3=relu3.gradient(gfc2)
        gfc1=fc1.gradient(grelu3)

        gpool2=pool2.gradient(gfc1)
        grelu2=relu2.gradient(gpool2)
        gbn2=bn2.backward(grelu2,lr=0.001)
        gconv2=conv2.gradient(gbn2)

        gpool1=pool1.gradient(gconv2)
        grelu1=relu1.gradient(gpool1)
        gbn1=bn1.backward(grelu1,lr=0.001)
        conv1.gradient(gbn1)
        # fc1.gradient(relu1.gradient(fc2.gradient(sf.eta)))

        
        
        fc2.backward(alpha=learning_rate, weight_decay=0.0004)
        fc1.backward(alpha=learning_rate, weight_decay=0.0004)

        # fc.backward(alpha=learning_rate, weight_decay=0.0004)
        conv2.backward(alpha=learning_rate, weight_decay=0.0004)
        conv1.backward(alpha=learning_rate, weight_decay=0.0004)

        mod = 10
        if i % mod == 0:
            print(time.time()-start)
            start=time.time()
            print ("i=%d   train_acc: %.4f  " % (i, train_acc / (mod * batch_size)))
            train_acc = 0

        if (i+1) % 100==0:
            test()
