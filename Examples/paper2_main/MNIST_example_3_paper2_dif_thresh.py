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
from Network.tools import U_thesh_coef, U_thesh_coef1

n_neurons_out = 20  # number of neurons in input layer
n_neurons_in = 196  # number of output in input layer
n_train =800# number of images for training
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
                                      U_tr=20/100,
                                      U_rest=0,
                                      refr_time=5,
                                      traces=True,
                                      inh=True)  # activate literal inhibition

conn = Connections(n_neurons_in, n_neurons_out, "all_to_all", w_min=0.00005, w_max=0.01)
conn.all_to_all_conn()
conn.initialize_weights("normal")

#print(out_neurons.U_thresh_all_neurons)
#coef0 , coef1= U_thesh_coef1()
coef0=torch.tensor([66.7521, 67.6613, 66.8345, 68.8402, 66.2031, 65.7780, 64.2742, 67.6287,
        63.5950, 64.0485, 64.1613, 63.3175, 63.4535, 64.5462, 59.9706, 62.3498,
        58.8332, 68.0912, 64.9158, 62.1731, 67.1491, 63.7720, 66.3503, 66.7815,
        67.1802, 66.3748, 66.2373, 63.0592, 63.8902, 60.8493, 64.4619, 65.8821,
        65.8412, 66.1914, 66.3855, 65.5294, 65.8859, 67.9008, 66.5146, 65.3538,
        66.0920, 66.0371, 66.7534, 68.3254, 66.0404, 66.8176, 67.2102, 68.5515,
        68.9180, 67.7385], dtype=torch.float64)
coef0=torch.tensor([21.8821, 23.2137, 24.5991, 26.0377, 27.5289, 29.0716, 30.6643, 32.3054,
        33.9930, 35.7247, 37.4978, 39.3096, 41.1567, 43.0357, 44.9428, 46.8739,
        48.8247, 50.7909, 52.7676, 54.7501, 56.7334, 58.7124, 60.6818, 62.6366,
        64.5715, 66.4811, 68.3604, 70.2042, 72.0073, 73.7650, 75.4722, 77.1245,
        78.7172, 80.2460, 81.7068, 83.0956, 84.4089, 85.6430, 86.7948, 87.8612,
        88.8395, 89.7271, 90.5218, 91.2216, 91.8247, 92.3296, 92.7350, 93.0400,
        93.2437, 93.3457], dtype=torch.float64)

coef0=torch.tensor([ 32.3721,  35.1571,  38.1034,  41.2104,  44.4761,  47.8974,  51.4698,  ##0.5
         55.1875,  59.0433,  63.0288,  67.1344,  71.3495,  75.6624,  80.0606,
         84.5308,  89.0591,  93.6313,  98.2329, 102.8492, 107.4654, 112.0671,
        116.6400, 121.1700, 125.6436, 130.0480, 134.3705, 138.5995, 142.7238,
        146.7328, 150.6168, 154.3666, 157.9737, 161.4304, 164.7294, 167.8641,
        170.8286, 173.6174, 176.2255, 178.6486, 180.8825, 182.9239, 184.7695,
        186.4166, 187.8629, 189.1062, 190.1448, 190.9774, 191.6028, 192.0202,
        192.2290], dtype=torch.float64)
coef0=torch.tensor([ 29.1613,  31.4543,  33.8687,  36.4038,  39.0583,  41.8299,  44.7154,
         47.7107,  50.8110,  54.0103,  57.3020,  60.6786,  64.1318,  67.6528,
         71.2321,  74.8596,  78.5251,  82.2176,  85.9264,  89.6402,  93.3480,
         97.0387, 100.7013, 104.3251, 107.8995, 111.4144, 114.8599, 118.2267,
        121.5058, 124.6887, 127.7673, 130.7342, 133.5823, 136.3049, 138.8961,
        141.3503, 143.6622, 145.8272, 147.8410, 149.6997, 151.3999, 152.9385,
        154.3128, 155.5203, 156.5590, 157.4272, 158.1234, 158.6466, 158.9958,
        159.1706], dtype=torch.float64)

coef0=torch.tensor([18.4118, 19.1325, 18.7502, 19.9498, 19.7680, 21.1226, 23.7044, 22.2646,
        24.6019, 23.5106, 25.7043, 24.6020, 24.6278, 25.9831, 24.8357, 28.3037,
        27.0011, 27.4581, 28.4245, 28.6087], dtype=torch.float64)
coef1=torch.mean(coef0)
out_neurons.U_thresh_all_neurons= torch.div(out_neurons.U_thresh_all_neurons, coef0)

new_U_thresh_all_neurons=out_neurons.U_thresh_all_neurons.clone().detach()
#print(out_neurons.U_thresh_all_neurons)
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
#plt.ion()
# x = list(range(400))
# y = np.random.randint(20, 40, 400)
# fig, ax = plt.subplots(figsize=(8, 5))
# line1, = ax.plot(x, y)
for i in range(n_train):
    print("Image № ", i, "of ", n_train)
    count1+=1
    if count1%20 == 0:

        fig.savefig("weights"+str(i))

    if data_train[i][1] in train_labels:
        input_spikes = encoding_to_spikes(data_train[i][0], time)
        # print(conn.weights)


        for j in tqdm(range(time),leave=False):

            out_neurons.compute_U_mem(input_spikes[j].reshape(196), conn.weights,crossbar=True,r_line=1)


            out_neurons.check_spikes(U_mem_all_neurons_decrease= 5/100/coef1*2, U_thresh_increase=0.2/100/(coef1)/2,
                                     diff_U_thresh=(True,new_U_thresh_all_neurons))
            """
            out_neurons.check_spikes(U_mem_all_neurons_decrease=5/100 , U_thresh_increase=0.2/100,
                                     diff_U_thresh=(True, new_U_thresh_all_neurons),
                                     diff_thresh_increase=(True,coef0.flipud()),diff_U_mem_all_neurons_decrease=(True,coef0))
            """
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
        """
        coef0 *= 0.99
        for hi, h in enumerate(coef0):
            if h < 18:
                coef0[hi] = 18
            

        out_neurons.U_thresh_all_neurons = torch.div(out_neurons.U_thresh_all_neurons, coef0)

        new_U_thresh_all_neurons = out_neurons.U_thresh_all_neurons.clone().detach()
        """
        print(coef0)

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
fig.savefig("spikes")



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
                out_neurons.check_spikes(U_mem_all_neurons_decrease=5 / 100 / coef1,
                                         U_thresh_increase=0.2 / 100 / coef1,
                                         diff_U_thresh=(True, new_U_thresh_all_neurons))
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