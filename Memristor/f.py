import torch

from Network.datasets import MNIST_train_test_14x14, encoding_to_spikes
from compute_crossbar import TransformToCrossbarBase

data_train = MNIST_train_test_14x14()[0]
input_spikes = encoding_to_spikes(data_train[0][0], 2)
f = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/weights_tensor.pt")
c=torch.matmul(f.reshape(1, 196), w)
print(input_spikes[1])
print(c)
g = TransformToCrossbarBase(w, 1000, 25000,1)
print(g.k)
print(g.G_min)
g.compute_crossbar(f)
g.plot_crossbar_U(f)
g.plot_crossbar_weights()
print(g.I_out)
print(torch.max(g.weights))
print(torch.div(torch.tensor(g.I_out), c)*100)
