import numpy as np
from LSTM.lstm import LstmParam, LstmNetwork,Loss
import pickle as pk

def train():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    np.random.seed(2)

    T = 100
    L = 1000
    N = 500

    x = np.empty((N, L), 'int64')
    t=np.arange(N)
    np.random.shuffle(t)
    x[:] = np.array(range(L)) +t.reshape(N, 1)
    data = np.sin(x / T).astype('float64')

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 1
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    loss=Loss(mem_cell_ct)
    epoch=1000
    for cur_iter in range(epoch):
        for n in range(N):
            input_val_arr = data[n, :-1]
            y_list =data[n, 1:]
            print("iter", "%2s" % str(cur_iter), end=": ")
            for ind in range(len(y_list)):
                lstm_net.x_list_add(input_val_arr[ind])

            lossv = lstm_net.y_list_is(y_list, loss)
            print("loss:", "%.3e" % lossv)
            lstm_param.apply_diff(lr=0.1)
            lstm_net.x_list_clear()
        if cur_iter % 10 ==0:
            fs = open('model%d.pkl'%cur_iter, 'wb')
            pk.dump(lstm_param,fs)
            pk.dump(loss,fs)
            fs.close()

if __name__ == "__main__":
    train()