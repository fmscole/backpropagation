from FCNet.dense import Dense
from FCNet.relu import *
from FCNet.my_read_Data import my_data_set
from FCNet.batch_normal import BatchNormal
class NetWork:
    def __init__(self):
        self.ds1=Dense(784,400)
        self.bn1=BatchNormal()
        self.at1=Relu()
        self.ds2=Dense(400,10)
        self.bn2=BatchNormal()
        self.at2 = sigmoid()
        self.n=0
        self.m = 0
    def forward(self,inputs,trainning=True):
        out=self.ds1.forward(inputs)
        out=self.bn1.forward(out,training=trainning)
        out=self.at1.forward(out)
        out=self.ds2.forward(out)
        out=self.bn2.forward(out,training=trainning)
        # out=self.at2.forward(out)
        out=softmax(out)
        return out
    def backward(self,eta):
        g=np.copy(eta)
        g=self.bn2.backward(g,lr=0.1)
        g=self.ds2.backward(eta=g,lr=0.1)
        g=self.at1.backward(g)
        g=self.bn1.backward(g,lr=0.1)
        g=self.ds1.backward(g,lr=0.1)

    def train(self,data,target):
        batch_size = data.shape[1]
        final_outputs=self.forward(data)
        eta=target-final_outputs

        self.n = self.n + 1
        # 处理批量准确率计算
        for i in range(batch_size):
            t = (final_outputs[:, i].argmax() == target[:, i].argmax())
            if t: self.m = self.m + 1

        # 每cm批次算一次准确率
        cm = 500
        if self.n % cm == 0:
            cc = self.m / (batch_size * cm)
            print(self.n, cc)
            self.m = 0


        self.backward(eta)
    def test(self):
        batch_size = 10
        data = my_data_set(kind='test')
        m=0
        n=0
        for k in range(10000//batch_size):

            # 小批量训练
            imgs, target = data.next_batch(batch_size)
            final_outputs = self.forward(imgs,trainning=False)
            # 处理批量准确率计算
            for i in range(batch_size):
                t = (final_outputs[:, i].argmax() == target[:, i].argmax())
                n = n + 1
                if t: m =m + 1

            # 每cm批次算一次准确率



        cc = m /n
        print("test:", cc)




n=NetWork()
data=my_data_set( kind='train')

batch=170  #np.min([i+10,100])
epochs = 10
for e in range(epochs):
    for i  in range(60000//batch):
        #小批量训练
        
        imgs,labs=data.next_batch(batch)
        n.train(imgs, labs)

        # if (i+1)%500==0:
    print("epochs=",e)
    n.test()
