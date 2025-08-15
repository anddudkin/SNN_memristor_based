import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt

from Network.visuals import plot_weights_square
def g(x):
    if x < 0.00005:
        return 1 / 0.000005
    else:
        return 1 / x
torch.set_printoptions(threshold=10_000)

# Applied voltages in volts.
applied_voltages = np.ones([5, 1])
#applied_voltages = np.ones([196, 1])
w = [
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100],
]
w1 = [
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100],
    [100, 100, 3000, 100, 100],
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100],
]
# #w = torch.load("C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/Examples/SNN_tests/weights_tensor.pt")
# #w1= torch.load("C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/Examples/SNN_tests/weights_tensor.pt")
# w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/70_3000/weights_tensor.pt")
# w1 = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/70_3000/weights_tensor.pt")
# w.apply_(g)
# w1.apply_(g)
# w=w.numpy()
# w1=w1.numpy()
# k=0
# for i in range(len(w1)):
#     for j in range(len(w1[0])):
#         if j !=0 and k < 500:
#             w1[i][j] = np.min(w1)
#         k+=1
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
aximW1 = axW1.imshow(w, cmap='gray_r', interpolation='None', vmin=0,
                     vmax=np.max(w1))
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
plt.show()

breakpoint()
print(applied_voltages)
w = torch.load("C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/Examples/SNN_tests/weights_tensor.pt")
# w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/70_3000/weights_tensor.pt")
print(torch.matmul(torch.ones([196]), w))
fig = plt.figure(figsize=(6, 6))
axV = fig.add_subplot(131)
ax1V = fig.add_subplot(132)
ax2V = fig.add_subplot(133)
ax = fig.add_subplot(231)
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)

axim = ax.imshow(plot_weights_square(196, 50, w), cmap='YlOrBr', vmin=0, vmax=1, interpolation='None')
plt.colorbar(axim, ax=ax, fraction=0.045, pad=0.04)
axim2 = ax1.imshow(w, cmap='YlOrBr', vmin=0, vmax=1, interpolation='None')
plt.colorbar(axim2, ax=ax1, fraction=0.12, pad=0.04)

# Device resistances in ohms.
resistances = w

# Interconnect resistance in ohms.
r_i = 1


# plt.imshow(w, cmap='gray_r', vmin=0, vmax=torch.max(w))
# plt.show()
def g(x):
    if x < 0.00005:
        return 1 / 0.000005
    else:
        return 1 / x


# Computing the solution.
# w = torch.clamp(w, min=0.99)
w.apply_(g)
# plt.imshow(w, cmap='gray', vmin=0, vmax=torch.max(w))
# plt.show()

solution = badcrossbar.compute(applied_voltages, w, r_i)
print(solution.currents.device)
axim1 = ax2.imshow(solution.currents.device, cmap='gray_r', interpolation='None')
# axim1 = ax2.imshow(solution.voltages.word_line, cmap='gray_r', vmin=0, vmax=1, interpolation='None')
plt.colorbar(axim1, ax=ax2, fraction=0.12, pad=0.04)
plt.tight_layout()
plt.show()
d = plt.imshow(w, cmap='gray', vmin=float(torch.min(w)), vmax=float(torch.max(w)), interpolation='None')
plt.colorbar(d)
plt.show()
