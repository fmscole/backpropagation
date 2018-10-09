from __future__ import print_function
import torch as t
from torch.autograd import Variable as V
def f(x):
    '''计算y'''
    y = x**2 * t.exp(x)
    return y

def gradf(x):
    '''手动求导函数'''
    dx = 2*x*t.exp(x) + x**2*t.exp(x)
    return dx
x = V(t.randn(3,4), requires_grad = True)
y = f(x)
print(y)

y.backward(t.ones(y.size())) # grad_variables形状与y一致
print(x.grad)
print(gradf(x) )