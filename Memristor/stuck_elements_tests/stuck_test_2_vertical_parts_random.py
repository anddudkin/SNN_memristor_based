import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

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
# w = torch.load("../Examples/SNN_tests/weights_tensor.pt")
# w1 = torch.load("../Examples/SNN_tests/weights_tensor.pt")

# w = torch.load("/home/anddudkin/PycharmProjects/SNN_memristor_based/Examples/SNN_tests/weights_tensor.pt")
# w1 = torch.load("/home/anddudkin/PycharmProjects/SNN_memristor_based/Examples/SNN_tests/weights_tensor.pt")

# w = torch.load("/home/anddudkin/PycharmProjects/SNN_memristor_based/Examples/mnist_example/weights_tensor.pt")
# w1 = torch.load("/home/anddudkin/PycharmProjects/SNN_memristor_based/Examples/mnist_example/weights_tensor.pt")

w = torch.load("C:/Users/anddu/OneDrive/Документы/GitHub/anddudkin_mem_project/Examples/SNN_tests/weights_tensor.pt")
w1 = torch.load("C:/Users/anddu/OneDrive/Документы/GitHub/anddudkin_mem_project/Examples/SNN_tests/weights_tensor.pt")
w.apply_(g)
w1.apply_(g)
w = w.numpy()
w1 = w1.numpy()
k = 0
probabil=0.005
n_neurons1=50
mask = np.random.binomial(n=1, p=probabil, size=[196,n_neurons1])
print(np.sum(mask)/196/n_neurons1 * 100)
for i in range(196):
    for j in range(n_neurons1):
        if mask[i][j] == 1:
            w1[i][j] = np.max(w)
probabil1=0.005
mask1 = np.random.binomial(n=1, p=probabil1, size=[196,n_neurons1])
for i in range(196):
    for j in range(n_neurons1):
        if mask1[i][j] == 1:
            w1[i][j] = np.min(w)

#w1[196 - (i * 14 + 14): 196 - i * 14, 0:50] = mask
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
mask1 = fig.add_subplot(3, 3, 3)
mask1.set_title("Mask " + str(probabil))
axV.set_title('Voltage')
ax.set_title('Current')
ax1V.set_title('Voltage')
ax1.set_title('Current')
ax2V.set_title('delta abs(V)')
ax2.set_title('delta I')
#mask
mask1_=mask1.imshow(mask, cmap='gray', interpolation='None', vmin=0,)
plt.colorbar(mask1_, ax=mask1, fraction=0.046, pad=0.04)
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
plt.subplots_adjust(bottom=0.07, left=0, right=0.85, top=0.965, wspace=0)

diff = np.abs(solution1.currents.output-solution.currents.output) /solution.currents.output * 100
print(solution1.currents.output)
print(solution.currents.output)
print(diff)
print("Mean", np.mean(diff))
print("min", np.min(diff))
print("max", np.max(diff))
print("Std", np.std(diff))
fig1 = plt.figure(figsize=(5, 5))
bar = fig1.add_subplot(3, 1, 1)
bar1 = fig1.add_subplot(3, 1, 2)
bar2 = fig1.add_subplot(3, 1, 3)
b=bar.imshow(solution.currents.output, cmap='gray_r', interpolation='None')
plt.colorbar(b, ax=bar, orientation='horizontal', shrink=0.5)
b1=bar1.imshow(solution1.currents.output, cmap='gray_r', interpolation='None')
plt.colorbar(b1, ax=bar1, orientation='horizontal', shrink=0.5)
b2 = bar2.imshow(diff, cmap='gray_r', interpolation='None')
plt.colorbar(b2, ax=bar2, orientation='horizontal', shrink=0.5)
fig1.tight_layout()
plt.show()
# fig.savefig("V_I"+str(i))
# fig1.savefig("I_out" + str(i))

with open('result'+'.txt', 'a+') as f:
    f.write("\nMean " + str(np.mean(diff)))
    f.write("\nMax " + str(np.max(diff)))
    f.write("\nMin " + str(np.min(diff)))
