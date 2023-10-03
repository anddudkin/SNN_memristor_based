import torch

from anddudkin_mem_project.visuals import plot_U_mem
from topology import construct_matrix_connections, inicialize_weights
from datasets import MNIST_train_test, rand_in_U
from NeuronModels import Neuron_IF
import matplotlib.pyplot as plt
n_neurons_out = 5
n_neurons_in = 20
conn = inicialize_weights(construct_matrix_connections(n_neurons_in, n_neurons_out, "all_to_all"))

data_train = MNIST_train_test()[0]
data_test = MNIST_train_test()[1]

print(conn)

out_neurons = Neuron_IF(n_neurons_out,n_neurons_in, 0, 10, -1, 5)

out_neurons.initialization()
for i in range(50):
    out_neurons.compute_U_mem(rand_in_U(n_neurons_in),conn)
    out_neurons.check_spikes()
    out_neurons.update_w(conn)
print(conn)
print("Umem\n",out_neurons.U_mem_all_neurons)
print("spikes\n",out_neurons.check_spikes())
print("refr\n",out_neurons.refractor_count)
print("spikes_trace_in\n",out_neurons.spikes_trace_in)
print("spikes_trace_out\n",out_neurons.spikes_trace_out)
print("dw\n",out_neurons.dw_all)
# for idx, img in enumerate(data_train):
#     for i in range(1):
#         pass

print(out_neurons.U_mem_trace)
plot_U_mem(n_neurons_out,out_neurons.U_mem_trace)
# print(img[0])
# print(img[0].view(784))
# print(img[0].view(28, 28))
# plt.imshow(torch.squeeze(img[0]),cmap='gray')
# plt.title(img[1])
# plt.show()


