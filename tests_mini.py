import torch

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

g = torch.tensor([[0,3,4,3]])
g1 = torch.tensor([[0,3,4,3]])
g2 = torch.tensor([[0,3,4,3]])
f=torch.cat([g,g1])
f2=torch.cat([f,g2])
print(g.shape)
print(f)
print(f2.shape)
print(f.shape)




#print(g[:,3])  возврящает столбцы очень удобно