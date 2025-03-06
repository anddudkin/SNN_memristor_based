import torch
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt

n_neurons_out = 50  # number of neurons in input layer
n_neurons_in = 196  # number of output in input layer
n_train =1# number of images for training
n_test = 1 # number of images for testing
time = 300  # time of each image presentation during training
time_test = 200 # time of each image presentation during testing
test = False # do testing or not
plot = True # plot graphics or not
# модель с реальными физическими величинами и линейно
# дискретным диапазоном значений проводимости и падением напряжений в кроссбар-массиве
out_neurons = NeuronLifAdaptiveThresh(n_neurons_in,
                                      n_neurons_out,
                                      train=True,
                                      U_mem=0,
                                      decay=0.92,
                                      U_tr=20/6500,
                                      U_rest=0,
                                      refr_time=5,
                                      traces=True,
                                      inh=True)  # activate literal inhibition

conn = Connections(n_neurons_in, n_neurons_out, "all_to_all", w_min=0.00005, w_max=0.01)
conn.all_to_all_conn()
conn.initialize_weights("normal")
data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]
train_labels = [0, 1, 2, 5, 9,3,4,6,7,8]
conn.load_weights('weights_tensor.pt')
plt.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
plt.show()
plt.imshow(conn.weights, cmap='YlOrBr', vmin=0.00005, vmax=0.01)
plt.show()

for i in tqdm(range(n_test), desc='test', colour='green', position=0):

    if data_train[i][1] in train_labels:
        input_spikes = encoding_to_spikes(data_train[i][0], time_test)

        for j in range(1):
            out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights)
            g=out_neurons.I_for_each_neuron

print(g)


for i in tqdm(range(n_test), desc='test', colour='green', position=0):

    if data_train[i][1] in train_labels:
        input_spikes = encoding_to_spikes(data_train[i][0], time_test)

        for j in range(1):
            out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights,crossbar=True,r_line=1)
            g1=out_neurons.I_for_each_neuron
print(g1)

f = torch.div(g,g1)
print(f)