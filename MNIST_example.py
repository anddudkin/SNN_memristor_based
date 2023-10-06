import torch

from anddudkin_mem_project.visuals import plot_U_mem
from topology import construct_matrix_connections, inicialize_weights, conn_matrix_transform
from datasets import MNIST_train_test, rand_in_U, encoding_to_spikes
from NeuronModels import Neuron_IF
import matplotlib.pyplot as plt

n_neurons_out = 20
n_neurons_in = 784
n_train = 2
n_test = 100
time = 5
conn = inicialize_weights(construct_matrix_connections(n_neurons_in, n_neurons_out, "all_to_all"))
conn1=conn_matrix_transform(conn,n_neurons_in,n_neurons_out)
data_train = MNIST_train_test()[2]
data_test = MNIST_train_test()[3]

print(conn)

out_neurons = Neuron_IF(n_neurons_out, n_neurons_in, 0, 80, -1, 5)
out_neurons.initialization()

for i in range(n_train):
    print("n_train.....",i,"/",n_train)
    input_spikes = encoding_to_spikes(data_train[i][0], time)
    for j in range(time):
        out_neurons.compute_U_mem_new(input_spikes[j].reshape(784), conn1)
        out_neurons.check_spikes()
        out_neurons.update_w(conn)
    plot_U_mem(n_neurons_out, out_neurons.U_mem_trace)
print(conn)
print("Umem\n", out_neurons.U_mem_all_neurons)
print("spikes\n", out_neurons.check_spikes())
print("refr\n", out_neurons.refractor_count)
print("spikes_trace_in\n", out_neurons.spikes_trace_in)
print("spikes_trace_out\n", out_neurons.spikes_trace_out)
print("dw\n", out_neurons.dw_all)
# for idx, img in enumerate(data_train):
#     for i in range(1):
#         pass

print(out_neurons.U_mem_trace)

# print(img[0])
# print(img[0].view(784))
# print(img[0].view(28, 28))
# plt.imshow(torch.squeeze(img[0]),cmap='gray')
# plt.title(img[1])
# plt.show()
