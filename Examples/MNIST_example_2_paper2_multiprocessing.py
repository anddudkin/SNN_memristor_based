import torch
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt
import time as t
import multiprocessing
def snn(ar):
    n_neurons_out = 50  # number of neurons in input layer
    n_neurons_in = 196  # number of output in input layer
    n_train =2000# number of images for training
    n_test = 1000# number of images for testing
    time = 350  # time of each image presentation during training
    time_test = 200 # time of each image presentation during testing
    test = True  # do testing or not
    plot = False # plot graphics or not
    # модель с реальными физическими величинами и линейно
    # дискретным диапазоном значений проводимости
    out_neurons = NeuronLifAdaptiveThresh(n_neurons_in,
                                          n_neurons_out,
                                          train=True,
                                          U_mem=0,
                                          decay=0.92,
                                          U_tr=20/100,
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



    if plot:
        plt.ion()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        axim = ax.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
        plt.colorbar(axim, fraction=0.046, pad=0.04)

        fig1 = plt.figure(figsize=(5, 5))
        ax2 = fig1.add_subplot(211)
        ax3 = fig1.add_subplot(212)
        axim2 = ax2.imshow(torch.zeros([14, 14]), cmap='gray', vmin=0, vmax=1, interpolation='None')
        axim3 = ax3.imshow(torch.zeros([196, 350])[::4, ::4], cmap='gray', vmin=0, vmax=1, interpolation='None')

    train_labels = [0, 1, 2, 9, 5]

    for i in tqdm(range(n_train)):

        if data_train[i][1] in train_labels:
            input_spikes = encoding_to_spikes(data_train[i][0], time)

            if plot:
                axim.set_data(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights))
                axim2.set_data(torch.squeeze(data_train[i][0]))
                axim3.set_data(input_spikes.reshape(196, 350)[::4, ::4])
                fig.canvas.flush_events()

            for j in range(time):
                out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights)
                out_neurons.check_spikes1()
                assig.count_spikes_train(out_neurons.spikes, data_train[i][1])
                conn.update_w2(out_neurons.spikes_trace_in, out_neurons.spikes_trace_out, out_neurons.spikes,
                               0.00005,0.01,ar, nonlinear=False)


    assig.get_assignment()
    #assig.save_assignment(path='assignments'+str(ar)+'.pkl')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    axim = ax.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
    plt.colorbar(axim, fraction=0.046, pad=0.04)
    fig.savefig("weights"+str(ar))
    evall = MnistEvaluation(n_neurons_out)

    #conn.save_weights(path='weights_tensor'+str(ar)+'.pt')
    #out_neurons.save_U_thresh(path='thresh'+str(ar)+'.pt')

    out_neurons.train = False
    out_neurons.reset_variables(True, True, True)

    if test:
        for i in tqdm(range(n_test), desc='test', colour='green', position=0):

            if data_train[i][1] in train_labels:
                input_spikes = encoding_to_spikes(data_train[i][0], time_test)

                for j in range(time_test):
                    out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights)
                    out_neurons.check_spikes1()
                    evall.count_spikes(out_neurons.spikes)
                evall.conclude(assig.assignments, data_train[i][1])

    evall.final()
    with open('result'+str(ar)+'.txt', 'w+') as f:
        f.write("\ntrain: " + str(train_labels))
        f.write("\ntrain: " + str(n_train))
        f.write("\ntest: " + str(n_test))
        f.write("\ntime_train: " + str(time))
        f.write("\ntime_train: " + str(time_test))
        f.write("\nneurons out: " + str(n_neurons_out))
        f.write("\ntrain images: " + str(0))

        f.write(evall.final())

if __name__ == "__main__":
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(snn, [16,32,64,128,256,512])
