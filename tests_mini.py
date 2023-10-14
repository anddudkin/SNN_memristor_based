import time

import torch

from anddudkin_mem_project.NeuronModels import NeuronLIF

"""
g = torch.tensor([[0,0,0.5],[0,1,0.7],
                  [1,0,0.8],[1,1,0.3],
                  [2,0,0.4],[2,1,0.5]])


print("g,shape  ", g.shape)
print(g)
b = g[:,2].reshape(3,2)
print("b===", b)
print("b shape", b.shape)

gg=torch.tensor([1,0,1],dtype=torch.float)
print("gg shape", gg.shape)
h=torch.matmul(gg,b)
print("hhhhhhhh ",h)

"""
b = NeuronLIF(100,200,0,decay=0.9)
print(b.traces)

"""
g = torch.Tensor([2,3])
g=g.repeat(3,1)

y = torch.Tensor([4,5,0])
print(y.reshape(3,1))
print(torch.sub(g,y.reshape(3,1)))
print(g)

n = torch.sub(g,y.reshape(3,1))
n.apply_(compute_dw)
print(n)
"""
"""
conn = Connections(9, 3, "all_to_all")
conn.all_to_all_conn()
conn.inicialize_weights()
b=conn.weights
print(conn.weights)
a1 = b[:,0].reshape(3,3)
a2 = b[:,1].reshape(3,3)
a3= b[:,2].reshape(3,3)
nn=[]
for i in range(3):
    nn.append(b[:,i].reshape(3,3))
c=torch.cat(nn,1)

print(c)
"""