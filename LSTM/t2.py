import numpy as np

from lstm import LstmParam, LstmNetwork


class ToyLossLayer:
    def __init__(self,mem_cell_ct):
        self.v=np.zeros(mem_cell_ct)
    
    def value(self, pred):
        out=self.v.dot(pred)  
        return out
    def loss(self, pred, label):
        out=self.value(pred)  
        return (out- label) ** 2


    def bottom_diff(self, pred, label):
        out=self.value(pred)
        df = 2 * (out - label)/self.v.shape[0] 
        diff=df*self.v
        self.v-=1.1*pred*df
        return diff


def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50

    y_list =  [-0.8333333333, 0.33333, 0.166666667, -80.8]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    lstm_param2 = LstmParam(mem_cell_ct, mem_cell_ct)
    lstm_net2 = LstmNetwork(lstm_param2)  

    loss=ToyLossLayer(mem_cell_ct)

    for cur_iter in range(2000):
        # print(y_list)
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            lstm_net2.x_list_add(lstm_net.lstm_node_list[ind].state.h)

        print("y_pred = [" +
              ", ".join(["% 2.5f" % loss.value(lstm_net2.lstm_node_list[ind].state.h) for ind in range(len(y_list))]) +
              "]", end=", ")

        lossv=lstm_net2.y_list_is(y_list, loss)
        lstm_net.y_list_is2(lstm_net2)
        print("loss:", "%.3e" % lossv)
        lstm_param2.apply_diff(lr=0.1)
        lstm_param.apply_diff(lr=0.1)
        lstm_net2.x_list_clear()
        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0()