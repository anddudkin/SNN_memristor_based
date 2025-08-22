import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14

plt.rcParams['axes.grid'] = True
figure = plt.figure(figsize=(10, 10))
f=False
axs = figure.subplots(
  nrows=3,
  ncols=3,
  sharex=f,
  sharey=f)

percents = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

k=1
for i in range(3):
    for j in range(3):
        sol_mean_all = np.load('data_mean' +str(k)+'.npy')  # load
        err_all = np.load('data_std' +str(k)+'.npy')  # load
        axs[i, j].plot(percents, sol_mean_all / 300, 'b-', linewidth=1)
        axs[i, j].errorbar(percents, sol_mean_all / 300, err_all / 300, fmt='s', markersize=4, capsize=4,linewidth=0.6)
        k += 1

for i in range(3):
    for j in range(3):
        axs[i, j].set_ylim(bottom=0)

axs[2,1].set_xlabel("Stuck elements, %",size=14)
axs[1,0].set_ylabel("Deviation, %",size=14)
plt.show()