import sys

import torch
from tqdm import tqdm
from tqdm import trange
from visuals import plot_U_mem, plot_weights
from topology import Connections
from datasets import MNIST_train_test, rand_in_U, encoding_to_spikes, MNIST_train_test_9x9, MNIST_train_test_14x14
from NeuronModels import NeuronIF, NeuronLIF, NeuronInhibitory
import matplotlib.pyplot as plt

n_neurons_out = 5
n_neurons_in = 196
n_train = 500
n_test = 100
time = 200
test = False
conn = Connections(n_neurons_in, n_neurons_out, "all_to_all")
conn.all_to_all_conn()
conn.initialize_weights("normal")

data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]

out_neurons = NeuronLIF(n_neurons_in, n_neurons_out, decay=0.94, U_tr=10, U_rest=-10, refr_time=2, traces=True)
inh_neurons = NeuronInhibitory(n_neurons_out,10)
plt.ion()
fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for i in tqdm(range(n_train), desc='Outer Loop', colour='green', position=0):
    if data_train[i][1] == 7 or data_train[i][1] == 1 or data_train[i][1] == 8:
        input_spikes = encoding_to_spikes(data_train[i][0], time)
        for j in range(time):
            out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights)
            out_neurons.check_spikes()
            print(out_neurons.spikes)
            out_neurons.U_mem_all_neurons=inh_neurons.compute_inhibition(out_neurons.spikes,out_neurons.U_mem_all_neurons)
            # print("spikes_trace_in\n", out_neurons.spikes_trace_in)
            # print("spikes_trace_out\n", out_neurons.spikes_trace_out)
            if torch.sum(out_neurons.spikes_trace_out) != 0:
                conn.update_w(out_neurons.spikes_trace_in, out_neurons.spikes_trace_out)

            # ax1 = fig1.add_subplot(111)
            # ax1.matshow(input_spikes.reshape(196, 30), cmap="gray")
        b = plot_weights(n_neurons_in, n_neurons_out, conn.weights)

        ax1.matshow(b, cmap='YlOrBr')
        ax2.imshow(torch.squeeze(data_train[i][0]), cmap='gray')
        plt.draw()
        plt.pause(0.005)

    # plot_U_mem(n_neurons_out, out_neurons.U_mem_trace)

    # plot_U_mem(n_neurons_out, out_neurons.U_mem_trace)
    # plt.show()
    # plt.pause(1)

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
