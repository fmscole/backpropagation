import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lstm import LstmParam, LstmNetwork,Loss
import pickle as pk

def test():
    fl = open('model200.pkl', 'rb')
    lstm_param=pk.load(fl)
    loss=pk.load(fl)
    fl.close()

    lstm_net = LstmNetwork(lstm_param)
    
    L = 100
    T=4
    F=100

    x= np.array(range(L))
    input_val_arr =0.5*np.sin(x / T).astype('float64')
    y_list =input_val_arr
    L=len(input_val_arr)
    
    for i in range(F):
        
        for ind in range(L):
            lstm_net.x_list_add(input_val_arr[ind])
        
        y=loss.value(lstm_net.lstm_node_list[L-1].state.h)
        input_val_arr=np.hstack((input_val_arr,[y]))
        input_val_arr=input_val_arr[1:]
        lstm_net.x_list_clear()
        y_list=np.hstack((y_list,[y]))
        # print(i,end=" ")

    print("here1")
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(yi, color):
        plt.plot(np.arange(L), yi[:L], color, linewidth = 2.0)
        plt.plot(np.arange(L, L+F), yi[L:], color + ':', linewidth = 2.0)
    draw(y_list, 'r')
    
    plt.savefig(r'predict%d-%d.pdf'%(L,F))
    plt.show()
    plt.close()

if __name__ == "__main__":
    test()