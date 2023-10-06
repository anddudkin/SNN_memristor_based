import torch

from anddudkin_mem_project.datasets import encoding_to_spikes
from anddudkin_mem_project.topology import inicialize_weights, construct_matrix_connections, conn_matrix_transform

# from visuals import DrawNN
# from datasets import test_values
# from compute_crossbar import compute_ideal
# from NeuronModels import Neuron_IF
# network = DrawNN([4, 10])
# network.draw()
# g = Neuron_IF(40,10,10,10,10)
# g.initialization()
# x=compute_ideal(test_values()[0],test_values()[1])[0]
# print(g.compute_U_mem(x)[1])
# print()

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

conn = inicialize_weights(construct_matrix_connections(4, 2, "all_to_all"))
conn_w=conn_matrix_transform(conn,4,2)

print(conn)
print(conn_w)


#print(g[:,3])  возврящает столбцы очень удобно