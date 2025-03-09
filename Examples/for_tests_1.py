import torch
from numpy.ma.core import shape
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt

n_neurons_out = 20  # number of neurons in input layer
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
out_neurons2 = NeuronLifAdaptiveThresh(n_neurons_in,
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

#conn.load_weights('../Examples/paper2_2/256_72/weights_tensor.pt')
#conn.load_weights('../Examples/paper2_2/20_neurons/weights_tensor.pt')
#conn.weights=torch.ones([196,50])/60
plt.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
plt.show()
plt.imshow(conn.weights, cmap='YlOrBr', vmin=0.00005, vmax=0.01)
plt.show()

out_neurons.compute_U_mem(torch.ones(196), conn.weights)
g=out_neurons.I_for_each_neuron
print(g)

out_neurons2.compute_U_mem(torch.ones(196), conn.weights,crossbar=True,r_line=1)
g1=out_neurons2.I_for_each_neuron
print(g1)

f = torch.div(g,g1)
print(f)
print(torch.mean(f))
plt.imshow(torch.unsqueeze(f,0), cmap='YlOrBr', vmin=min(f), vmax=max(f))
plt.show()
out_neurons3 = NeuronLifAdaptiveThresh(n_neurons_in,
                                      n_neurons_out,
                                      train=True,
                                      U_mem=0,
                                      decay=0.92,
                                      U_tr=20/6500,
                                      U_rest=0,
                                      refr_time=5,
                                      traces=True,
                                      inh=True)  # activate literal inhibition
out_neurons4 = NeuronLifAdaptiveThresh(n_neurons_in,
                                      n_neurons_out,
                                      train=True,
                                      U_mem=0,
                                      decay=0.92,
                                      U_tr=20/6500,
                                      U_rest=0,
                                      refr_time=5,
                                      traces=True,
                                      inh=True)  # activate literal inhibition


g5=0

out_neurons3.reset_variables(True,True,True)
for i in tqdm(range(1000)):
    input_spikes = encoding_to_spikes(data_train[i][0], time_test)
    out_neurons3.compute_U_mem(input_spikes[0].reshape(196), conn.weights)
    g5+=out_neurons3.I_for_each_neuron
    out_neurons3.reset_variables(True,True,True)
g4=0
for i in tqdm(range(1000)):
    input_spikes = encoding_to_spikes(data_train[i][0], time_test)
    out_neurons3.compute_U_mem(input_spikes[0].reshape(196), conn.weights,crossbar=True,r_line=1)
    g4+=out_neurons3.I_for_each_neuron
    out_neurons3.reset_variables(True, True, True)
f1 = torch.div(g5/300,g4/300)
print(f1)

