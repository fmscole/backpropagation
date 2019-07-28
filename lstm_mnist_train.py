import numpy as np
from LSTM.lstm import Lstm_cell
import pickle as pk

from layers.my_read_Data import my_data_set

data = my_data_set(kind='train')
test_data = my_data_set(kind='test')

    
def softmax(x):
    v = np.exp(x - x.max(axis=-1, keepdims=True))    
    return v / v.sum(axis=-1, keepdims=True)
    
        
class Loss:
    def __init__(self,mem_cell_ct):
        self.weights = np.random.normal(0.0, pow(100, -0.5), (100,10))
        self.total=0
        self.acc=0
        self.i=0
    def value(self, pred):
        self.x=pred
        pred=pred.copy()
        pred=np.dot(pred,self.weights)
        pred=softmax(pred)
        return pred
    def loss(self, pred, label):
        
        pred=self.value(pred)
        t=np.argmax(pred)==np.argmax(label)
        if(t):
            self.acc+=1
            # print(np.sum((pred- label) ** 2))
        else:
            a=1
        self.i+=1
        self.total+=1
        if self.total>100:
            print(self.i,"-------------------------------------",self.acc/self.total)
            self.acc=0
            self.total=0
        # if self.acc/self.total<0.9:
        #     print("-------------------------------------",self.acc/self.total)
        return np.sum((pred- label) ** 2)
    def bottom_diff(self, pred, label):
        pred=self.value(pred)
        dy=pred-label
        dw=np.outer(self.x,dy)
        dy=np.dot(dy,self.weights.T)
        self.weights-=0.05*dw
        return dy

def train():
    T = 28
    L = 28
    N = 100

    # x = np.empty((N, L), 'int64')
    # t=np.arange(N)
    # np.random.shuffle(t)
    # x[:] = np.array(range(L)) +t.reshape(N, 1)
    # data = np.sin(x / T).astype('float64')

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 28

    lstm_net = Lstm_cell(mem_cell_ct, x_dim)
    loss=Loss(mem_cell_ct)
    epoch=5*60000//N
    for cur_iter in range(epoch):
        # print(cur_iter,"-----------------------------------------------------------------------------------------")
        batch_size=N
        imgs, labs = test_data.next_batch(batch_size)
        imgs=imgs.reshape(-1,28,28)
        for n in range(N):
            input_val_arr=imgs[n]
            # =np.repeat(labs[n].reshape(1,10),28,axis=0)
            label=labs[n]

            # print("iter", "%2s" % str(cur_iter), end=": ")
            for ind in range(28):
                pred=lstm_net.x_list_add(input_val_arr[ind])
            loss.loss(pred,label)
            dh=loss.bottom_diff(pred, label)
            # lossv = lstm_net.y_list_is(y_list, loss)
            # print("loss:", "%.3e" % lossv)
            # lstm_param.apply_diff(lr=0.1)
            lstm_net.backward(dh)
            lstm_net.apply_diff(lr=0.01)
        # if cur_iter % 10 ==0:
        #     fs = open('model%d.pkl'%cur_iter, 'wb')
        #     pk.dump(lstm_param,fs)
        #     pk.dump(loss,fs)
        #     fs.close()

if __name__ == "__main__":
    train()