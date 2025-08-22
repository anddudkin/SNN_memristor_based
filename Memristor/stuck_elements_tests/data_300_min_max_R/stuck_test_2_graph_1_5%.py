import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14


figure = plt.figure(figsize=(6, 13))

axs = figure.subplots(
  nrows=2,
  ncols=3,
  sharex=True,
  sharey=True)

percents = [ 1, 2, 3, 4, 5, 6 ]

k=0
for i in range(2):
    for j in range(3):
        mask = np.random.binomial(n=1, p=percents[k]/100, size=[196, 50])
        axs[i, j].imshow(mask,cmap="gray",)
        axs[i, j].set_title(str(percents[k])+"%")
        k += 1

plt.subplots_adjust(bottom=0.07, left=0, right=0.2, top=0.98, wspace=0)
plt.tight_layout()
plt.subplots_adjust( hspace=0.03,bottom=0.27,)
plt.show()

