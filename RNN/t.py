import numpy as np
input_dim=10
hidden_nodes=200
output_dim=10
x=np.random.random([input_dim])
Wxh = np.random.random([hidden_nodes, input_dim])*0.01
Bxh = np.random.random([hidden_nodes])*0.01
Whh = np.random.random([hidden_nodes, hidden_nodes])*0.01
Bhh = np.random.random([hidden_nodes])*0.01
Wyh = np.random.random([output_dim, hidden_nodes])*0.01
Byh = np.random.random([output_dim])*0.01
h = np.random.random([hidden_nodes])*0.01
# print (np.matmul(x,x))
# print (Wxh.shape)
print(Wxh/np.sqrt(Wxh*Wxh))