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


def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if value < array[0]:
        return array[0]
    elif value > array[n - 1]:
        return array[-1]
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    print(jl)
    print(value - array[jl])
    if value == array[0]:  # edge cases at bottom
        return array[0]

    elif (value - array[jl]) <= (array[jl+1] - value):
        return array[jl]
    else:
        return array[jl+1]


print(bisection([1,2,3,4,5,6,7], 7))
breakpoint()
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
