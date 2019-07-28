import random

import numpy as np
import math

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values*(1-values)

def tanh_derivative(values): 
    return 1. - values ** 2

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args): 
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 

    def apply_diff(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo) 
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo) 

class LstmState:
    def __init__(self,batch_size, mem_cell_ct):
        self.g = np.zeros((batch_size,mem_cell_ct))
        self.i = np.zeros((batch_size,mem_cell_ct))
        self.f = np.zeros((batch_size,mem_cell_ct))
        self.o = np.zeros((batch_size,mem_cell_ct))
        self.s = np.zeros((batch_size,mem_cell_ct))
        self.h = np.zeros((batch_size,mem_cell_ct))
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
    
class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def bottom_data_is(self, x, s_prev = None, h_prev = None):
        # if this is the first lstm node in the network
        
        if s_prev is None: s_prev = np.zeros(self.state.s.shape)
        if h_prev is None: h_prev = np.zeros(self.state.h.shape)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        self.state.g = np.tanh(np.tensordot(xc,self.param.wg,axes=(1,1)) + self.param.bg)
        self.state.i = sigmoid(np.tensordot(xc,self.param.wi,axes=(1,1)) + self.param.bi)
        self.state.f = sigmoid(np.tensordot(xc,self.param.wf,axes=(1,1)) + self.param.bf)
        self.state.o = sigmoid(np.tensordot(xc,self.param.wo,axes=(1,1)) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc

        # return self.state.h,self.state.s
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di 
        df_input = sigmoid_derivative(self.state.f) * df 
        do_input = sigmoid_derivative(self.state.o) * do 
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.dot(di_input.T, self.xc)
        self.param.wf_diff += np.dot(df_input.T, self.xc)
        self.param.wo_diff += np.dot(do_input.T, self.xc)
        self.param.wg_diff += np.dot(dg_input.T, self.xc)
        self.param.bi_diff +=np.sum(di_input,axis=0)
        self.param.bf_diff += np.sum(df_input,axis=0)       
        self.param.bo_diff += np.sum(do_input,axis=0)
        self.param.bg_diff += np.sum(dg_input,axis=0)       

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(di_input,self.param.wi)
        dxc += np.dot(df_input,self.param.wf)
        dxc += np.dot(do_input,self.param.wo)
        dxc += np.dot(dg_input,self.param.wg)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[:,self.param.x_dim:]
        self.state.bottom_diff_x = dxc[:,:self.param.x_dim]

class Lstm_cell():
    def __init__(self, mem_cell_ct, x_dim):
        self.lstm_param = LstmParam(mem_cell_ct, x_dim)
        self.lstm_node_list = []
        # input sequence
        self.x_list = []
    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(x.shape[0], self.lstm_param.mem_cell_ct)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)
        return self.lstm_node_list[idx].state.h

    
    def backward(self,diff_h):
        idx = len(self.x_list) - 1
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros((diff_h.shape[0],self.lstm_param.mem_cell_ct))
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            # diff_h =0 loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1 
    def apply_diff(self,lr=0.1):
        self.lstm_param.apply_diff(lr)
        self.x_list_clear()


    

