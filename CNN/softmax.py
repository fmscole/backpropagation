import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
class Softmax(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batchsize = shape[0]

    def cal_loss(self, prediction, label):
        # self.label = label
        # self.prediction = prediction
        # self.predict(prediction)
        self.softmax=self.predict(prediction)
        # self.softmax=sigmoid(prediction)
        self.eta=self.softmax-label
        # self.loss = 0
        # for i in range(self.batchsize):
        #     self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

        # return self.loss

    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)

        exp_prediction=prediction.copy()
        for i in range(self.batchsize):
            exp_prediction[i, :] -= np.max(exp_prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        
        return self.softmax

    def gradient(self):
        # self.eta = self.softmax.copy()
        # for i in range(self.batchsize):
        #     self.eta[i, self.label[i]] -= 1
        
        return self.eta
