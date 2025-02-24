import math

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

print(np.linspace(0,4,5))
print(np.linspace(4,8,5))
x1=np.concatenate((np.linspace(0,4,5)[:-1],np.linspace(4,8,5)))
print(x1)
def linespace_diff_dens(start, end, num1, num2, s):
    num1+=1
    l_all = end - start
    first_point = l_all * s
    return np.concatenate((np.linspace(start, first_point, num1)[:-1],np.linspace(first_point,end,num2)))

g = linespace_diff_dens(0.00005,0.01,56,200,0.5)
print(len(g))

plt.loglog( list(range(1,257)),g, 'o',markersize=2)
plt.show()
breakpoint()

x = np.linspace(0.00005, 0.01, 50)
y_ = 1 / 3 ** (-x * 300)
plt.plot(y_, x, 'o')
plt.show()

breakpoint()
with open('result.txt', 'a+') as f:
    f.write("\ntrain: ")
    f.write("\ntrain: ")
    f.write("\ntrain: ")
data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]
print(type(data_test))
for i in range(10):
    print(data_train[i][1])
for i in range(10):
    print(data_train[i][1])
breakpoint()
x = []
for i in np.linspace(0.00005, 0.01, 20000):
    x.append(1 / i)
plt.plot(np.linspace(0.00005, 0.01, 20000), x)
plt.show()

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
