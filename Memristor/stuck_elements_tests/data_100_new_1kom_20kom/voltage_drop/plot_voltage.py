import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt



applied_voltages = np.ones([196, 1])
# applied_voltages = np.ones([196, 1])



torch.set_printoptions(threshold=10_000)

fig = plt.figure(figsize=(4, 6))
# w =np.full((196, 50), 10000)
w1=np.full([196, 50], 1000)
w1= np.random.randint(1, 20, size=(196, 50))
y1 = plt.imshow(w1, cmap="gray", interpolation='None')
x1=plt.colorbar(y1)
x1.set_label(label = "Сопротивление, кОм",  fontsize=18, rotation=270, labelpad=25)
x1.ax.tick_params(labelsize=17)
plt.xlabel("Номер столбца", fontsize=18)
plt.ylabel("Номер строки", fontsize=18)
# plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.show()


solution = badcrossbar.compute(applied_voltages, w1, 10)
y = plt.imshow(solution.voltages.word_line, cmap="gray", vmin=0, vmax=1, interpolation='None')
x=plt.colorbar(y)
x.set_label(label = "Напряжение, В",  fontsize=18, rotation=270, labelpad=25)
x.ax.tick_params(labelsize=17)

plt.xlabel("Номер столбца", fontsize=18)
plt.ylabel("Номер строки", fontsize=18)
# plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig("fig0")
plt.tight_layout()
plt.show()
