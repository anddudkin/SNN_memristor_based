import pickle

import torch
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh

import matplotlib.pyplot as plt
from Memristor import compute_crossbar

n_neurons_out = 50  # number of neurons in input layer
n_neurons_in = 196  # number of output in input layer
n_train = 5  # number of images for training
n_test = 500 # number of images for testing
time = 350  # time of each image presentation during training
time_test = 10# time of each image presentation during testing
test = True  # do testing or not
plot = False  # plot graphics or not

out_neurons = NeuronLifAdaptiveThresh(n_neurons_in,
                                      n_neurons_out,
                                      train=False,
                                      U_mem=0,
                                      decay=0.92,
                                      U_tr=2,
                                      U_rest=0,
                                      refr_time=5,
                                      traces=True,
                                      inh=True)  # activate literal inhibition

conn = Connections(n_neurons_in, n_neurons_out, "all_to_all")
conn.all_to_all_conn()
conn.initialize_weights("normal")

data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]

assig = MnistAssignment(n_neurons_out)
assig.load_assignment('assignments.pkl')

train_labels = [0, 1, 2, 9]

evall = MnistEvaluation(n_neurons_out)
torch.set_printoptions(threshold=10_000)
conn.load_weights('weights_tensor.pt')
out_neurons.load_U_thresh('thresh.pt')

from Memristor.compute_crossbar import TransformToCrossbarBase

with open('Res_states.pkl', 'rb') as f:
    r = pickle.load(f)
cbw = TransformToCrossbarBase(conn.weights, 5000, 25000, 1)
cbw.transform_with_experemental_data(r)
# cbw.plot_crossbar_weights()
out_neurons.train = False
out_neurons.reset_variables(True, True, True)
count2 = 0
if test:
    for i in tqdm(range(n_test), desc='test', colour='green', position=0):
        print(i)
        if data_train[i][1] in train_labels:
            input_spikes = encoding_to_spikes(data_train[i][0], time_test)
            out_neurons.reset_variables(True, True, True)
            for j in tqdm(range(time_test),desc='U_mem', colour='green', position=0):
                out_neurons.compute_U_mem(input_spikes[j] / 2, cbw.weights_Om, k=100000, crossbar=True, r_line=1,
                                          nonlin=True)
                out_neurons.check_spikes()

                # if 1 in out_neurons.spikes:
                #     print(out_neurons.spikes)
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
    f.write("\ntest images: " + str(count2))
    f.write(evall.final())
