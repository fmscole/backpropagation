__author__ = 'qixianbiao'
#A toy example for RNN.
#It can be run in CPU because of the small scale.
import numpy as np
#错误：这里应该是softmax，而不是sigmoid
#sigmoid(x)=1/(np.exp(-x)+1)
def sigmoid(x):
    x = (np.exp(x)+0.00000000001)/np.sum(np.exp(x)+0.00000000001)
    return x
#print(sigmoid([1,2]))
class RNN:
    def __init__(self, input_dim, hidden_nodes, output_dim):
        self.Wxh = np.random.random([hidden_nodes, input_dim])*0.01
        self.Bxh = np.random.random([hidden_nodes])*0.01
        self.Whh = np.random.random([hidden_nodes, hidden_nodes])*0.01
        self.Bhh = np.random.random([hidden_nodes])*0.01
        self.Wyh = np.random.random([output_dim, hidden_nodes])*0.01
        self.Byh = np.random.random([output_dim])*0.01
        self.h = np.random.random([hidden_nodes])*0.01

    def forward(self, x):
        T = x.shape[1]
        states = []
        output = []
        for i in range(T):
            if i == 0:
                ht = np.tanh(np.dot(self.Wxh, x[:, i]) + self.Bxh + np.dot(self.Whh, self.h))

            else:
                ht = np.tanh(np.dot(self.Wxh, x[:, i]) + self.Bxh + np.dot(self.Whh, states[i-1]))
            ot = sigmoid(np.dot(self.Wyh, ht) + self.Byh)
            states.append(ht)
            output.append(ot)
        return states, output

    def backword(self, x, y, h, output, lr=0.002):
        T = x.shape[1]
        dL_T = np.dot( np.transpose(self.Wyh), output[-1]-y[:, -1])
        loss = np.sum(-y[:, -1]*np.log(output[-1]))
        dL_ht = dL_T
        D_Wyh = np.zeros_like(self.Wyh)
        D_Byh = np.zeros_like(self.Byh)
        D_Whh = np.zeros_like(self.Whh)
        D_Bhh = np.zeros_like(self.Bhh)
        D_Wxh = np.zeros_like(self.Wxh)
        D_Bxh = np.zeros_like(self.Bxh)
        for t in range(T-2, -1, -1):
            dQ = (output[t] - y[:, t])
            DL_Qt = np.dot(np.transpose(self.Wyh), dQ)

            dy = (1 - h[t]*h[t])
            dL_ht += np.dot(np.transpose(self.Wyh), dQ)

            D_Wyh += np.outer(dQ, h[t])
            D_Byh += dQ

            D_Wxh += np.outer(dy*dL_ht, x[:, t])
            D_Bxh += dy*dL_ht

            D_Whh += np.outer(dy*dL_ht, h[t-1])
            D_Bhh += dy*dL_ht

            loss += np.sum(-y[:, t]*np.log(output[t]))
        for dparam in [D_Wyh, D_Byh, D_Wxh, D_Bxh, D_Whh, D_Bhh]:
            np.clip(dparam, -5, 5, out=dparam)

        self.Wyh -= lr*D_Wyh#/np.sqrt(D_Wyh*D_Wyh + 0.00000001)
        self.Wxh -= lr*D_Wxh##/np.sqrt(D_Wxh*D_Wxh + 0.00000001)
        self.Whh -= lr*D_Whh#/np.sqrt(D_Whh*D_Whh + 0.00000001)
        self.Byh -= lr*D_Byh#/np.sqrt(D_Byh*D_Byh + 0.00000001)
        self.Bhh -= lr*D_Bhh#/np.sqrt(D_Bhh*D_Bhh + 0.00000001)
        self.Bxh -= lr*D_Bxh#/np.sqrt(D_Bxh*D_Bxh + 0.00000001)
        self.h -= lr*dL_ht#/np.sqrt(dL_ht*dL_ht + 0.00000001)
        return loss, self.h
    def sample(self, x):
        h = self.h
        predict = []
        for i in range(9-1):
            ht = np.tanh(np.dot(self.Wxh, x) + self.Bxh + np.dot(self.Whh, h))
            ot = sigmoid(np.dot(self.Wyh, ht) + self.Byh)
            ynext = np.argmax(ot)
            predict.append(ynext)
            x = np.zeros_like(x)
            x[ynext] = 1
        return predict










