import numpy
import copy
def softmax(x):
    #减去最大值
    t=copy.copy(x)
    t-=numpy.max(t)
    t=numpy.exp(t)
    s=numpy.sum(t)
    t = t/s
    return t
    
a=numpy.array(range(12)).reshape(3,4).T
# b=numpy.array([a[:,i] for i in range(4)])
f =numpy.array([softmax(a[:,i]) for i in range(a.shape[1])]).T
print(a)
print(f)