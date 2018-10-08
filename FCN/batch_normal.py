import numpy as np
class BatchNormal:
    def __init__(self):
        self.gamma=1
        self.beta=0
        self.epsilon=1e-5
        pass
    def forward(self,input,axis=0):
        self.m=input.shape[axis]
        mu=np.sum(input,axis=axis,keepdims=True)/self.m
        # print(mu)
        self.xmu=input-mu
        # print(self.xmu)
        var=self.xmu**2
        var = np.sum(var)/self.m+self.epsilon
        self.ivar=1/np.sqrt(var)
        self.xhut=self.xmu*self.ivar
        return self.gamma*self.xhut+self.beta
    def backward(self,dy,lr=0.001):
        dbeta=np.sum(dy)
        self.beta-=dbeta
        dgmama=np.sum(dy*self.xhut)
        self.gamma-=dgmama

        dxhut=dy*self.gamma
        # depsilon=-0.5*np.sum(dxhut*self.xmu*np.power(self.ivar,3))
        # self.epsilon-=depsilon


        dx=dxhut*self.ivar*(1-(1+(self.xmu*self.ivar)**2)/self.m)

        return dx

if __name__=="__main__":
    x=np.array(np.arange(16)).reshape((2,2,4))
    nb=BatchNormal()
    # print(x)
    print(nb.forward(x))

    # print(nb.backward(x))