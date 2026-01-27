import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt



applied_voltages = np.ones([196, 1])
# applied_voltages = np.ones([196, 1])



torch.set_printoptions(threshold=10_000)


# w =np.full((196, 50), 10000)
w1=np.full([196, 50], 1000)
w = np.random.randint(1000, 20000, size=(196, 50))

fig = plt.figure(figsize=(4, 6))
solution = badcrossbar.compute(applied_voltages, w1, 2)
plt.imshow(solution.voltages.word_line,cmap='Grays', vmin=0, vmax=1, interpolation='None')
plt.colorbar()


plt.xlabel("Column number", fontsize=20)
plt.ylabel("Row number", fontsize=20)
# plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig("fig0")

plt.show()
