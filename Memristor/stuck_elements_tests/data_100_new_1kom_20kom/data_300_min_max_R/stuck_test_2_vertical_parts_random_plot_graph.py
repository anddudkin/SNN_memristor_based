import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt




sol_mean_all= np.load('data_mean.npy') # load
err_all=np.load('data_std.npy') # load
percents = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
plt.plot(percents, sol_mean_all / 40, 'b-', linewidth=1)
plt.errorbar(percents, sol_mean_all / 40, err_all / 40, fmt='s', markersize=4, capsize=4,linewidth=0.6)
plt.xlabel("Stuck elements, %")
plt.ylabel("Deviation, %")
plt.grid(True)
plt.ylim(bottom=0)

plt.show()