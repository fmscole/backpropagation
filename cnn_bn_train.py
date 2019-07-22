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
import time

data = my_data_set(kind='train')
test_data = my_data_set(kind='test')

batch_size = 167

conv1 = Conv2D([batch_size,28, 28,1], 8, 5, 1)
bn1=BN()
relu1 = Relu()
pool1 = Pooling()
conv2 = Conv2D([batch_size,24, 24,8], 16, 5, 1)
bn2=BN()
relu2 = Relu()
# pool2 = Pooling()
fc1 = FullyConnect(1024, 200)
relu3 = Relu()
fc2 = FullyConnect(200, 10)
sf = Softmax()

def  test():
    train_acc=0
    total=0
    batch_size=10000
    for i in range(10000//batch_size):
        imgs, labs = test_data.next_batch(batch_size)
        sf=forward(imgs, labs,training=False)
        # train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf[j]) == np.argmax(labs[j]):
                train_acc += 1
            total+=1
        # print ("%d /%d   train_acc: %.4f  " % (i,10000//batch_size//10,train_acc / total))
    print ("Test_acc: %.4f  " % (train_acc / total))


def forward(imgs, labs,training=True):
    conv1_out = conv1.forward(imgs)
    bn1_out=bn1.forward(conv1_out,training=training)
    relu1_out = relu1.forward(bn1_out)
    pool1_out = pool1.forward(relu1_out)

    conv2_out = conv2.forward(pool1_out)
    bn2_out = bn2.forward(conv2_out,training=training)
    relu2_out = relu2.forward(bn2_out)
    # pool2_out = pool2.forward(relu2_out)

    fc1_out1 = fc1.forward(relu2_out)
    relu3_out1 = relu3.forward(fc1_out1)
    fc2_out = fc2.forward(relu3_out1)
    sf_out=sf.forward(fc2_out)
    return sf_out


for epoch in range(5):
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
        sf_out=forward(imgs, labs)

        for j in range(batch_size):
            if np.argmax(sf_out[j]) == np.argmax(labs[j]):
                train_acc += 1

        # sf.gradient()
        gfc2=fc2.backward(sf_out-labs)
        grelu3=relu3.backward(gfc2)
        gfc1=fc1.backward(grelu3)

        # gpool2=pool2.backward(gfc1)
        grelu2=relu2.backward(gfc1)
        gbn2=bn2.backward(grelu2,lr=0.001)
        gconv2=conv2.backward(gbn2)

        gpool1=pool1.backward(gconv2)
        grelu1=relu1.backward(gpool1)
        gbn1=bn1.backward(grelu1,lr=0.001)
        conv1.backward(gbn1)
        
        
        fc2.gradient(alpha=learning_rate, weight_decay=0.001)
        fc1.gradient(alpha=learning_rate, weight_decay=0.001)

        
        conv2.gradient(alpha=learning_rate, weight_decay=0.001)
        conv1.gradient(alpha=learning_rate, weight_decay=0.001)

        mod = 100
        if i % mod == 0:
            
            print ("epoch=%d  batchs=%d   train_acc: %.4f  " % (epoch,i, train_acc / (mod * batch_size)))
            train_acc = 0

            print("----------------------------------------------------------------------------------------------------")
            print("epoch=",epoch,"     time(sec) is ",time.time()-start)
            start=time.time()
            test()
            print("----------------------------------------------------------------------------------------------------")
