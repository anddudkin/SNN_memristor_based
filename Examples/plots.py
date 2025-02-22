import numpy as np
import torch
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.auto import tqdm  # notebook compatible
import time


x = []
for i in np.linspace(0.00005, 0.01, 20000):
    x.append(1 / i)
plt.plot(np.linspace(0.00005, 0.01, 20000), x)
plt.show()
breakpoint()
n_neurons_out = 50  # number of neurons in input layer
n_neurons_in = 196

conn = Connections(n_neurons_in, n_neurons_out, "all_to_all")
conn.all_to_all_conn()
conn.initialize_weights("normal")

conn.load_weights('C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/Examples/paper2_1/weights_tensor.pt')
plt.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

conn.load_weights('C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/Examples/paper2_0/weights_tensor.pt')
plt.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0, vmax=1)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
