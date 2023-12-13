import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

from Network.visuals import plot_weights_square

torch.set_printoptions(threshold=10_000)
# Applied voltages in volts.
applied_voltages =np.ones([196, 1])
print(applied_voltages)
w=hh = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/70_3000/weights_tensor.pt")
print(torch.matmul(torch.ones([196]),w))
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(131)
ax1 = fig.add_subplot(132)
ax2 = fig.add_subplot(133)

axim = ax.imshow(plot_weights_square(196, 70, w), cmap='YlOrBr', vmin=0, vmax=1,interpolation='None')
plt.colorbar(axim, ax=ax,fraction=0.045, pad=0.04)
axim2=ax1.imshow(w, cmap='YlOrBr', vmin=0, vmax=1,interpolation='None')
plt.colorbar(axim2, ax=ax1,fraction=0.12, pad=0.04)


# Device resistances in ohms.
resistances = w

# Interconnect resistance in ohms.
r_i = 0.1


# plt.imshow(w, cmap='gray_r', vmin=0, vmax=torch.max(w))
# plt.show()
def g(x):
    if x == 0:
        return 1 / 0.00005
    else:
        return 1 / x


# Computing the solution.
# w = torch.clamp(w, min=0.99)
w.apply_(g)
# plt.imshow(w, cmap='gray', vmin=0, vmax=torch.max(w))
# plt.show()
print(torch.max(w))
solution = badcrossbar.compute(applied_voltages, w, r_i)
print(np.sum(solution.voltages.word_line) + np.sum(solution.voltages.bit_line))
axim1=ax2.imshow(solution.voltages.word_line, cmap='gray_r', vmin=0, vmax=1,interpolation='None')
plt.colorbar(axim1, ax=ax2,fraction=0.12, pad=0.04)
plt.tight_layout()
plt.show()

print(torch.tensor(solution.currents.output))


