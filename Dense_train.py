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

batch_size = 170

fc1 = FullyConnect(784, 200)
relu3 = Relu()
fc2 = FullyConnect(200, 10)
sf = Softmax()

def  test():
    train_acc=0
    total=0
    batch_size=17
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
    
    fc1_out1 = fc1.forward(imgs)
    relu3_out1 = relu3.forward(fc1_out1)
    fc2_out = fc2.forward(relu3_out1)
    sf_out=sf.forward(fc2_out)
    return sf_out


for epoch in range(50):
    start=time.time()
    learning_rate = 0.1

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

        
        
        fc2.gradient(alpha=learning_rate, weight_decay=0.001)
        fc1.gradient(alpha=learning_rate, weight_decay=0.001)

        
    print("epoch=",epoch,"----------------------------------------------------------------------------------------------------")
    print(time.time()-start)
    start=time.time()
    test()
    print("----------------------------------------------------------------------------------------------------")
