import torch
from numpy.ma.core import shape
from tqdm import tqdm
from Network.assigment import MnistAssignment, MnistEvaluation
from Network.visuals import plot_weights_square
from Network.topology import Connections
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14
from Network.NeuronModels import NeuronLifAdaptiveThresh
import matplotlib.pyplot as plt
# assig = MnistAssignment(25)
# assig.load_assignment("assignments.pkl")
# print(assig.assignments)

n_neurons_out = 20  # number of neurons in input layer
n_neurons_in = 196  # number of output in input layer
n_train =1# number of images for training
n_test = 1 # number of images for testing
time = 300  # time of each image presentation during training
time_test = 200 # time of each image presentation during testing
test = False # do testing or not
plot = True # plot graphics or not
o=torch.tensor([17.8152, 19.2839, 20.4578, 21.8190, 21.1654, 22.5894, 23.2838, 24.8887,
        26.4274, 28.0985, 26.6248, 27.9894, 30.1907, 32.5933, 33.8627, 32.3070,
        36.7874, 36.3242, 37.9317, 37.3736, 39.4426, 42.2269, 42.3704, 43.1366,
        45.9213, 45.3061, 49.0792, 49.5411, 48.4679, 50.6780, 53.4367, 53.9363,
        54.7987, 54.0393, 54.5429, 55.9020, 57.4862, 60.2043, 59.1948, 57.0785,
        59.2419, 61.3416, 63.4817, 61.3317, 61.8395, 63.7624, 64.7882, 62.5945,
        63.8072, 65.0278], dtype=torch.float64)
o1=torch.tensor([11.5404, 18.7387, 10.9946, 15.8173, 19.2851, 10.8467, 11.1172, 11.6756,
        12.0457, 11.8449, 18.5724, 22.7157, 14.6564, 15.1461, 18.9676, 16.0954,
        20.7554, 13.6225, 12.7118, 14.7965, 22.8652, 13.5021, 20.0030, 18.5032,
        13.4753, 16.8547, 14.7376, 20.3223, 23.4707, 12.7542, 13.3499, 16.4003,
        17.3758, 13.8427, 23.4175, 14.6691, 16.2786, 22.5335, 26.2085, 20.8227,
        24.7418, 26.3368, 14.7471, 17.3240, 25.7018, 21.0549, 20.1345, 33.0958,
        40.2239, 28.6554], dtype=torch.float64)
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

conn2 = Connections(n_neurons_in, n_neurons_out, "all_to_all", w_min=0.00005, w_max=0.01)
conn2.all_to_all_conn()
conn2.initialize_weights("normal")

data_train = MNIST_train_test_14x14()[0]
data_test = MNIST_train_test_14x14()[1]
input_spikes = encoding_to_spikes(data_train[0][0], time_test)
#out_neurons.compute_U_mem(input_spikes[0].reshape(196), conn.weights)
#conn.load_weights('../Examples/paper2_2/256_72/weights_tensor.pt')
#conn.load_weights('../Examples/paper2_2/20_neurons/weights_tensor.pt')
#conn.weights=torch.ones([196,50])/60
# plt.imshow(plot_weights_square(n_neurons_in, n_neurons_out, conn.weights), cmap='YlOrBr', vmin=0.00005, vmax=0.01)
# plt.show()
# plt.imshow(conn.weights, cmap='YlOrBr', vmin=0.00005, vmax=0.01)
# plt.show()
out_neurons.compute_U_mem(input_spikes[0].reshape(196), conn.weights)
g=out_neurons.I_for_each_neuron
print("нормальное распределение весов")
print(g)
out_neurons.reset_variables(True,True,True)
conn2.load_weights('../Examples/paper2_2/20_neurons/weights_tensor.pt')
out_neurons.compute_U_mem(input_spikes[0].reshape(196), conn2.weights)
# out_neurons.compute_U_mem(torch.ones(196), conn.weights)
g1=out_neurons.I_for_each_neuron
print("распределение весов обученной нейросети")
print(g1)
print("до обучения / после обучения")
print(torch.div(g,g1))
print("нормальное распределение весов + межсоединения")
out_neurons2.compute_U_mem(input_spikes[0].reshape(196), conn.weights,crossbar=True,r_line=1)
print(torch.mean(conn.weights))
print(torch.mean(conn2.weights))
g2=out_neurons2.I_for_each_neuron
print(g2)
out_neurons.reset_variables(True,True,True)
conn2.load_weights('../Examples/paper2_2/20_neurons/weights_tensor.pt')
out_neurons2.compute_U_mem(input_spikes[0].reshape(196), conn2.weights,crossbar=True,r_line=1)
g3=out_neurons2.I_for_each_neuron
print("распределение весов обученной нейросети + межсоединения")
print(g3)

print(torch.div(g,g2))
print(torch.div(g1,g3))
breakpoint()
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

