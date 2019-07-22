import numpy as np
from functools import reduce
class BatchNormal:
    def __init__(self):
        self.gamma=1
        self.beta=0.0
        self.epsilon=1e-5
        self.mean=None
        self.var=None
        pass
    def forward(self,input,axis=3,momentum=0.95,training=True):
        '''
        axis: channel所在的维度,比如input为[batch,height,width,channel],则axis=3（或-1）。
        这样就是对整个batch的同一特征平面（feature）标准化。
        不是针对每个样本标准化.也不是对每个特征平面标准化，而是把整个batch的同一个特征平面放在一起标准化.
        在求和求平均值的时候，channel维度保留，其他三个维度坍缩为一个数，塌缩到channel上。
        '''
        if training:
            shape=list(input.shape)
            ax=list(np.arange(len(shape)))
            shape.pop(axis)
            ax.pop(axis)
            self.axis=tuple(ax)
            self.m=reduce(lambda x, y: x * y, shape)


            mu=np.mean(input,axis=self.axis,keepdims=True)
            self.xmu=input-mu
            var = np.var(input,axis=self.axis,keepdims=True)
            self.ivar=1/np.sqrt(var+self.epsilon)
            self.xhut=self.xmu*self.ivar

            if self.mean is None: self.mean=mu
            if self.var is None: self.var =var

            self.mean=self.mean*momentum+mu*(1-momentum)
            self.var = self.var * momentum + var * (1 - momentum)

            return self.gamma*self.xhut+self.beta
        else:
            return self.test(input=input)
    def test(self,input):
        xmu = input - self.mean
        ivar = 1 / np.sqrt(self.var + self.epsilon)
        xhut = xmu * ivar

        return self.gamma*xhut+self.beta

    def backward(self,dy,lr=0.09):
        '''
        lr:学习率
        '''
        dxhut=dy*self.gamma
        dx1=self.m*dxhut
        dx2=self.ivar**2*np.sum(dxhut*self.xmu,axis=self.axis,keepdims=True)*self.xmu
        dx3=np.sum(dxhut,axis=self.axis,keepdims=True)
        dx=self.ivar/self.m*(dx1-dx2-dx3)

        dbeta=np.sum(dy,axis=self.axis,keepdims=True)
        self.beta-=lr*dbeta #根据dy的符号情况，有的网络这里的+要改为-
        dgmama=np.sum(dy*self.xhut,axis=self.axis,keepdims=True)
        self.gamma-=lr*dgmama  #根据dy的符号情况，有的网络这里的+要改为-

        return dx


if __name__=="__main__":
    x=np.array(np.arange(24)).reshape((3,2,2,2))
    nb=BatchNormal()
    print(x[0,0])
    print(nb.forward(x,axis=1)[0,0])

    dy=np.array([[[[1.3028, 0.5017],
       [-0.8432, -0.2807]],

      [[-0.4656, 0.2773],
       [-0.7269, 0.1338]]],

     [[[-3.1020, -0.7206],
       [0.4891, 0.2446]],

      [[0.2814, 2.2664],
       [0.8446, -1.1267]]],

     [[[-2.4999, 1.0087],
       [0.6242, 0.4253]],

      [[2.5916, 0.0530],
       [0.5305, -2.0655]]]])

    dnb=nb.backward(dy=dy)
    print(dnb[0,0])