import sys

import torch
from tqdm import tqdm
from assigment import MnistAssignment, MnistEvaluation
from visuals import plot_U_mem, plot_weights_square
from topology import Connections
from datasets import MNIST_train_test, rand_in_U, encoding_to_spikes, MNIST_train_test_9x9, MNIST_train_test_14x14
from NeuronModels import NeuronIF, NeuronLIF, NeuronInhibitory, NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt

n_neurons_out = 12
n_neurons_in = 196
n_train = 5
n_test = 5
time = 350
time_test = 200
test = True
conn = Connections(n_neurons_in, n_neurons_out, "all_to_all")
conn.all_to_all_conn()
conn.initialize_weights("normal")

data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]
train_labels = [0, 1, 9]

out_neurons = NeuronLifAdaptiveThresh(n_neurons_in,
                                      n_neurons_out,
                                      train=True,
                                      U_mem=0,
                                      decay=0.92,
                                      U_tr=20,
                                      U_rest=0,
                                      refr_time=5,
                                      traces=True,
                                      inh=True)

assig = MnistAssignment(n_neurons_out)
inh_neurons = NeuronInhibitory(n_neurons_out, 13)
plt.ion()
fig = plt.figure(figsize=(6, 6))
# fig1 = plt.figure(figsize=(5, 5))

ax1 = fig.add_subplot(111)
axim2 = ax1.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0, vmax=1)
plt.colorbar(axim2, fraction=0.046, pad=0.04)
fig.tight_layout()
# ax2 = fig1.add_subplot(211)
# ax3 = fig1.add_subplot(212)

for i in tqdm(range(n_train), desc='Outer Loop', colour='green', position=0):

    if data_train[i][1] in train_labels:
        input_spikes = encoding_to_spikes(data_train[i][0], time)

        axim2.set_data(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights))
        fig.canvas.flush_events()

        # ax2.imshow(torch.squeeze(data_train[i][0]), cmap='gray')
        # ax3.imshow(input_spikes.reshape(196, 350)[::4, ::4], cmap='gray', vmin=0, vmax=1)

        for j in range(time):
            out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights)

            out_neurons.check_spikes()
            assig.count_spikes_train(out_neurons.spikes, data_train[i][1])
            conn.update_w(out_neurons.spikes_trace_in, out_neurons.spikes_trace_out, out_neurons.spikes)

        # plot_U_mem(n_neurons_out, out_neurons.U_mem_trace)
        # plt.show()
        # plt.pause(1)
        # out_neurons.U_mem_trace = torch.zeros([1, n_neurons_out], dtype=torch.float)
    # plot_U_mem(n_neurons_out, out_neurons.U_mem_trace)
plt.close()
plt.ioff()

plt.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0, vmax=1)
plt.show()
assig.get_assigment()
evall = MnistEvaluation(n_neurons_out)
out_neurons.train = False

if test:
    for i in tqdm(range(n_test), desc='test', colour='green', position=0):

        if data_train[i][1] in train_labels:
            input_spikes = encoding_to_spikes(data_train[i][0], time_test)

            for j in range(time_test):
                out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights)
                out_neurons.check_spikes()
                evall.count_spikes(out_neurons.spikes)

            evall.conclude(assig.assignments, data_train[i][1])

evall.final()
