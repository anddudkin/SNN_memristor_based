import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

from Network.visuals import plot_weights_square
from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14

data_train = MNIST_train_test_14x14()[0]
input_spikes = encoding_to_spikes(data_train[0][0], 2)
applied_voltages = input_spikes[0].reshape(196, 1)
#applied_voltages = np.ones([196, 1])
def g(x):
    if x < 0.00005:
        return 1 / 0.000005
    else:
        return 1 / x


torch.set_printoptions(threshold=10_000)


# w = torch.load("C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/Examples/SNN_tests/weights_tensor.pt")
# w1= torch.load("C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/Examples/SNN_tests/weights_tensor.pt")
w = torch.load("../Examples/SNN_tests/weights_tensor.pt")
w1 = torch.load("../Examples/SNN_tests/weights_tensor.pt")
w.apply_(g)
w1.apply_(g)
w = w.numpy()
w1 = w1.numpy()
k = 0
mask= np.ones([20, 50]) * np.min(w1)
w1[80:100,0:50]= mask


#plt.show()

for i in range(0,196,14):

    mask = np.ones([14, 50]) * np.min(w1)
    w1[i:i+14, 0:50] = mask
    r_i = 1
    fig = plt.figure(figsize=(8, 8))
    axW1 = fig.add_subplot(3, 3, 1)
    axV = fig.add_subplot(3, 3, 4)
    ax1V = fig.add_subplot(3, 3, 5)
    ax2V = fig.add_subplot(3, 3, 6)
    axW2 = fig.add_subplot(3, 3, 2)
    ax = fig.add_subplot(3, 3, 7)
    ax1 = fig.add_subplot(3, 3, 8)
    ax2 = fig.add_subplot(3, 3, 9)
    axV.set_title('Voltage')
    ax.set_title('Current')
    ax1V.set_title('Voltage')
    ax1.set_title('Current')
    ax2V.set_title('delta abs(V)')
    ax2.set_title('delta I')
    # crossbars
    aximW1 = axW1.imshow(w, cmap='gray_r', interpolation='None', vmin=0,
                         vmax=np.max(w))
    plt.colorbar(aximW1, ax=axW1, fraction=0.046, pad=0.04)
    aximW2 = axW2.imshow(w1, cmap='gray_r', interpolation='None', vmin=0,
                         vmax=np.max(w1))
    plt.colorbar(aximW2, ax=axW2, fraction=0.046, pad=0.04)

    solution = badcrossbar.compute(applied_voltages, w, r_i)
    v = solution.voltages.word_line
    c = solution.currents.device
    aximV = axV.imshow(v, cmap='gray_r', interpolation='None', vmin=0,
                       vmax=np.max(v))
    plt.colorbar(aximV, ax=axV, fraction=0.046, pad=0.04)

    axim = ax.imshow(solution.currents.device, cmap='gray_r', interpolation='None', vmin=0,
                     vmax=np.max(solution.currents.device))
    plt.colorbar(axim, ax=ax, fraction=0.046, pad=0.04)

    solution1 = badcrossbar.compute(applied_voltages, w1, r_i)
    v1 = solution1.voltages.word_line
    c1 = solution1.currents.device
    axim1V = ax1V.imshow(v1, cmap='gray_r', interpolation='None', vmin=0,
                         vmax=np.max(v1))
    plt.colorbar(axim1V, ax=ax1V, fraction=0.046, pad=0.04)

    axim1 = ax1.imshow(solution1.currents.device, cmap='gray_r', interpolation='None', vmin=0,
                       vmax=np.max(solution1.currents.device))
    plt.colorbar(axim1, ax=ax1, fraction=0.046, pad=0.04)
    v3 = np.abs(v - v1)
    c3 = c - c1
    axim2V = ax2V.imshow(v3, interpolation='None')
    plt.colorbar(axim2V, ax=ax2V, fraction=0.046, pad=0.04)

    axim2 = ax2.imshow(c3, interpolation='None')
    plt.colorbar(axim2, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07, left=0, right=0.85, top=0.98, wspace=0)

    diff = solution1.currents.output / solution.currents.output * 100 - 100
    print(solution1.currents.output)
    print(solution.currents.output)
    print(diff)
    print("Mean", np.mean(diff))
    print("min", np.min(diff))
    print("max", np.max(diff))
    fig1 = plt.figure(figsize=(5, 5))
    bar = fig1.add_subplot(3, 1, 1)
    bar1 = fig1.add_subplot(3, 1, 2)
    bar2 = fig1.add_subplot(3, 1, 3)
    bar.imshow(solution.currents.output, cmap='gray_r', interpolation='None')
    bar1.imshow(solution1.currents.output, cmap='gray_r', interpolation='None')
    b1 = bar2.imshow(diff, cmap='gray_r', interpolation='None')
    plt.colorbar(b1, ax=bar2, orientation='horizontal', shrink=0.5)
    fig1.tight_layout()
    fig.savefig("V_I"+str(i))
    fig1.savefig("I_out" + str(i))
    with open('result'+'.txt', 'a+') as f:
        f.write("\n " + str(i))
        f.write("\nMean" + str(np.mean(diff)))
        f.write("\nMax" + str(np.max(diff)))
        f.write("\nMin" + str(np.min(diff)))
