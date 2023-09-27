import torch
from topology import construct_matrix_connections, inicialize_weights
from datasets import MNIST_train_test
from NeuronModels import Neuron_IF

conn = inicialize_weights(construct_matrix_connections(10, 5, "all_to_all"))

data_train = MNIST_train_test()[0]
data_test = MNIST_train_test()[1]

print(conn)

out_neurons = Neuron_IF(5, 0, 1, -1, 5)

out_neurons.initialization()

out_neurons.compute_U_mem([1,1,0,0,1],conn)

print("Umem",out_neurons.U_mem_all_neurons)
print("spikes",out_neurons.check_spikes())
print("refr",out_neurons.refractor_count)
out_neurons.compute_U_mem([1,1,1,0,1],conn)
print("Umem",out_neurons.U_mem_all_neurons)
print("spikes",out_neurons.check_spikes())
print("refr",out_neurons.refractor_count)
out_neurons.compute_U_mem([1,1,0,1,1],conn)
print("Umem",out_neurons.U_mem_all_neurons)
print("spikes",out_neurons.check_spikes())
print("refr",out_neurons.refractor_count)
out_neurons.compute_U_mem([1,1,1,0,1],conn)
out_neurons.compute_U_mem([1,1,1,0,1],conn)
out_neurons.compute_U_mem([1,1,1,0,1],conn)
out_neurons.compute_U_mem([1,1,1,0,1],conn)
print("Umem",out_neurons.U_mem_all_neurons)
print("spikes",out_neurons.check_spikes())
print("refr",out_neurons.refractor_count)
