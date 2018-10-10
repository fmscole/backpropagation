import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable as V

m = nn.BatchNorm2d(2,momentum=None,  affine=False)
x=np.array(np.arange(24)).reshape((3,2,2,2))
input=V(torch.Tensor(x),requires_grad = True)
print(input[0,0])
output = m(input)
print(output[0,0])
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

dy=torch.tensor(dy)

output.backward(dy)
print(input.grad[0,0])