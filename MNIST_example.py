import torch

from anddudkin_mem_project.visuals import plot_U_mem, plot_weights
from topology import Connections
from datasets import MNIST_train_test, rand_in_U, encoding_to_spikes, MNIST_train_test_9x9, MNIST_train_test_14x14
from NeuronModels import Neuron_IF
import matplotlib.pyplot as plt

n_neurons_out = 10
n_neurons_in = 196
n_train = 10
n_test = 100
time = 150
test = False
conn = Connections(n_neurons_in, n_neurons_out, "all_to_all")
conn.all_to_all_conn()
conn.inicialize_weights()

data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]

print(conn)

out_neurons = Neuron_IF(n_neurons_out, n_neurons_in, 0, 100, -1, 5, traces=True)
out_neurons.initialization()

plt.ion()
fig =plt.figure()

for i in range(n_train):
    print("n_train.....", i, "/", n_train)
    input_spikes = encoding_to_spikes(data_train[i][0], time)
    for j in range(time):
        out_neurons.compute_U_mem_new(input_spikes[j].reshape(196), conn.weights)
        print(input_spikes[j])
        out_neurons.check_spikes()
        print("spikes\n", out_neurons.spikes)
        # print("spikes_trace_in\n", out_neurons.spikes_trace_in)
        # print("spikes_trace_out\n", out_neurons.spikes_trace_out)
        conn.update_w(out_neurons.spikes_trace_in, out_neurons.spikes_trace_out)
        b = plot_weights(n_neurons_in, n_neurons_out, conn.weights)
        ax = fig.add_subplot(111)
        ax.matshow(b, cmap='YlOrBr')
        plt.draw()
        plt.pause(0.05)
        plt.clf()

    plot_U_mem(n_neurons_out, out_neurons.U_mem_trace)

if test:
    for i in range(n_test):
        print("n_test.....", i, "/", n_test)
        input_spikes = encoding_to_spikes(data_test[i][0], time)

print(conn)
print("Umem\n", out_neurons.U_mem_all_neurons)

print("refr\n", out_neurons.refractor_count)
print("spikes_trace_in\n", out_neurons.spikes_trace_in)
print("spikes_trace_out\n", out_neurons.spikes_trace_out)
print("dw\n", out_neurons.dw_all)
# for idx, img in enumerate(data_train):
#     for i in range(1):
#         pass


# print(img[0])
# print(img[0].view(784))
# print(img[0].view(28, 28))
# plt.imshow(torch.squeeze(img[0]),cmap='gray')
# plt.title(img[1])
# plt.show()
