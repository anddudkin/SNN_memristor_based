import json

import numpy as np
import torch
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt
import time as t
import pickle
n_neurons_out = 25 # number of neurons in input layer
n_neurons_in = 196  # number of output in input layer
n_train =600# number of images for training
n_test = 1000 # number of images for testing
time = 200  # time of each image presentation during training
time_test = 100 # time of each image presentation during testing
test = True  # do testing or not
plot = True# plot graphics or not
# модель с реальными физическими величинами и линейно
# дискретным диапазоном значений проводимости и падением напряжений в кроссбар-массиве
out_neurons = NeuronLifAdaptiveThresh(n_neurons_in,
                                      n_neurons_out,
                                      train=True,
                                      U_mem=0,
                                      decay=0.92,
                                      U_tr=20/100/28,
                                      U_rest=0,
                                      refr_time=5,
                                      traces=True,
                                      inh=True)  # activate literal inhibition

conn = Connections(n_neurons_in, n_neurons_out, "all_to_all", w_min=0.00005, w_max=0.01)
conn.all_to_all_conn()
conn.initialize_weights("normal")



data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]

assig = MnistAssignment(n_neurons_out)

num_spikes=[]

if plot:
    plt.ion()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    axim = ax.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
    plt.colorbar(axim, fraction=0.046, pad=0.04)

    # fig1 = plt.figure(figsize=(5, 5))
    # ax2 = fig1.add_subplot(211)
    # ax3 = fig1.add_subplot(212)
    # axim2 = ax2.imshow(torch.zeros([14, 14]), cmap='gray', vmin=0, vmax=1, interpolation='None')
    # axim3 = ax3.imshow(torch.zeros([196, time])[::4, ::4], cmap='gray', vmin=0, vmax=1, interpolation='None')
c=0
train_labels = [0, 1, 2, 9, 5]
count1=0
plt.ion()
x = list(range(400))
# y = np.random.randint(20, 40, 400)
# fig, ax = plt.subplots(figsize=(8, 5))
# line1, = ax.plot(x, y)
for i in range(n_train):
    print("Image № ", i, "of ", n_train)
    count1+=1
    if count1%20 == 0:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        axim = ax.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005,
                         vmax=0.01)
        plt.colorbar(axim, fraction=0.046, pad=0.04)
        fig.savefig("weights"+str(i))

    if data_train[i][1] in train_labels:
        input_spikes = encoding_to_spikes(data_train[i][0], time)
        # print(conn.weights)


        for j in tqdm(range(time),leave=False):

            out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights,crossbar=True,r_line=1)

            # if 1 in out_neurons.spikes:
            #      print(out_neurons.spikes)
            out_neurons.check_spikes(U_mem_all_neurons_decrease= 5/100/17, U_thresh_increase=0.2/100/20)
            c+=int(sum(out_neurons.spikes))

            assig.count_spikes_train(out_neurons.spikes, data_train[i][1])
            conn.update_w2(out_neurons.spikes_trace_in, out_neurons.spikes_trace_out,
                           out_neurons.spikes, 0.00005, 0.01, 128, nonlinear=True)

            if plot:
                axim.set_data(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights))

                # axim2.set_data(torch.squeeze(data_train[i][0]))
                # axim3.set_data(input_spikes.reshape(196, time)[::4, ::4])
                fig.canvas.flush_events()
        num_spikes.append(c)
        c=0
        print(num_spikes)
        # line1.set_xdata(list(range(len(num_spikes))))
        # line1.set_ydata(num_spikes)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
with open("spikes.json", 'w') as f1:
    # indent=2 is not needed but makes the file human-readable
    # if the data is nested
    json.dump(num_spikes, f1, indent=2)
#fig.savefig("spikes")



assig.get_assignment()
assig.save_assignment()


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
axim = ax.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
plt.colorbar(axim, fraction=0.046, pad=0.04)
fig.savefig("weights")
evall = MnistEvaluation(n_neurons_out)

conn.save_weights()
out_neurons.save_U_thresh()

out_neurons.train = False
out_neurons.reset_variables(True, True, True)

if test:
    for i in tqdm(range(n_test), desc='test', colour='green', position=0):

        if data_train[i][1] in train_labels:
            input_spikes = encoding_to_spikes(data_train[i][0], time_test)

            for j in range(time_test):
                out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights,crossbar=True,r_line=1)
                out_neurons.check_spikes(U_mem_all_neurons_decrease= 5/100/17, U_thresh_increase=0.2/100/20)
                evall.count_spikes(out_neurons.spikes)
            evall.conclude(assig.assignments, data_train[i][1])

evall.final()
with open('result.txt', 'w+') as f:
    f.write("\ntrain: " + str(train_labels))
    f.write("\ntrain: " + str(n_train))
    f.write("\ntest: " + str(n_test))
    f.write("\ntime_train: " + str(time))
    f.write("\ntime_train: " + str(time_test))
    f.write("\nneurons out: " + str(n_neurons_out))
    f.write("\ntrain images: " + str(0))

    f.write(evall.final())