import numpy as np
a=np.array([[1,2,3]])
print(np.diag(a)-np.dot(a.T,a))