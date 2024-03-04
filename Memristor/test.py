import badcrossbar
import numpy as np
import matplotlib.pyplot as plt
# Applied voltages in volts.
# 2D arrays must be used to avoid ambiguity.
applied_voltages = np.ones([196, 1])
n_out=100
# Device resistances in ohms.

resistances = np.random.randint(1000,25000,(196,n_out))
# Interconnect resistance in ohms.
r_i = 0.5

# Computing the solution.
solution = badcrossbar.compute(applied_voltages, resistances, r_i)
g=np.sum(solution.voltages.word_line)/196/n_out
print("min",(np.min(solution.voltages.word_line))*100)
print("mean",(1-g)*100)
print(g*100)

# plt.imshow(solution.voltages.word_line, cmap='gray_r', vmin=0, vmax=1, interpolation='None')
# plt.show()