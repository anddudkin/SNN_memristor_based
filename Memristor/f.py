import badcrossbar
import matplotlib.pyplot as plt
import torch

from Network.datasets import MNIST_train_test_14x14, encoding_to_spikes
from compute_crossbar import TransformToCrossbarBase
data_train = MNIST_train_test_14x14()[0]
input_spikes = encoding_to_spikes(data_train[0][0], 2)

w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/70_3000/weights_tensor.pt")
print(torch.matmul(input_spikes[1].reshape(1,196),w))
g = TransformToCrossbarBase(w, 100, 15000, 0.001)

g.compute_crossbar(input_spikes[1])
g.plot_crossbar_U(input_spikes[1])
g.plot_crossbar_weights()
print(g.I_out)