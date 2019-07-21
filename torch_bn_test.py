import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable as V
from CNN.batch_normal import BatchNormal as BN

bn1=BN()
m = nn.BatchNorm2d(2,momentum=None,  affine=False)
x=np.array(np.random.randint(-10,10,24)).reshape((3,2,2,2))
input=V(torch.Tensor(x),requires_grad = True)
print(input[0,0])
output = m(input)
output2 = bn1.forward(x,1)
print("torch---",output[0,0])
print("my------",output2[0,0])
# print(output.requires_grad)
dy=list([[[[ 1.3028,  0.5017],
          [-0.8432, -0.2807]],

         [[-0.4656,  0.2773],
          [-0.7269,  0.1338]]],


        [[[-3.1020, -0.7206],
          [ 0.4891,  0.2446]],

         [[ 0.2814,  2.2664],
          [ 0.8446, -1.1267]]],


        [[[-2.4999,  1.0087],
          [ 0.6242,  0.4253]],

         [[ 2.5916,  0.0530],
          [ 0.5305, -2.0655]]]])
dnb=bn1.backward(np.array(dy))
print("")
print("")
print("my---",dnb[0,0])
dy=torch.tensor(dy)

output.backward(dy)
print("torch--",input.grad[0,0])