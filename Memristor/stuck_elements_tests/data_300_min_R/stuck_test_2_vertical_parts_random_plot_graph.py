import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14

data_train = MNIST_train_test_14x14()[0]
input_spikes = encoding_to_spikes(data_train[0][0], 2)
applied_voltages = input_spikes[0].reshape(196, 1)


# applied_voltages = np.ones([196, 1])



sol_mean_all= np.load('data_mean.npy') # load
err_all=np.load('data_std.npy') # load
percents = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
plt.plot(percents, sol_mean_all / 300, 'b-', linewidth=1)
plt.errorbar(percents, sol_mean_all / 300, err_all / 300, fmt='s', markersize=4, capsize=4,linewidth=0.6)
plt.xlabel("Stuck elements, %")
plt.ylabel("Deviation, %")
plt.grid(True)
#plt.ylim([0, 45])
plt.show()
