import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14

data_train = MNIST_train_test_14x14()[0]
input_spikes = encoding_to_spikes(data_train[0][0], 2)
applied_voltages = input_spikes[0].reshape(196, 1)


# applied_voltages = np.ones([196, 1])



sol_mean_all= np.load('data_mean0.npy') # load
err_all=np.load('data_std0.npy') # load
percents = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
plt.plot(percents, sol_mean_all / 200, 'b-', linewidth=1)
plt.errorbar(percents, sol_mean_all / 200, err_all / 200, fmt='s', markersize=4, capsize=4,linewidth=0.6)
plt.xlabel("Stuck elements, %")
plt.ylabel("Deviation, %")
plt.grid(True)
#plt.ylim([0, 45])
plt.show()

figure = plt.figure(figsize=(10, 5))

axs = figure.subplots(
  nrows=3,
  ncols=3,
  sharex=True,
  sharey=True)




sol_mean_all= np.load('data_mean0.npy') # load
err_all=np.load('data_std0.npy') # load
k=1
for i in range(3):
    for j in range(3):
        axs[i, j].plot(percents, np.load('data_mean'+str(k)+'.npy')/300)
        axs[i, j].errorbar(percents, np.load('data_mean'+str(k)+'.npy')/300,
                           np.load('data_std'+str(k)+'.npy')/300, fmt='s', markersize=4, capsize=4,linewidth=0.6)
        k += 1
plt.show()