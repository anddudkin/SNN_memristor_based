import badcrossbar
import matplotlib.pyplot as plt

applied_voltages = [
    [1],
    [1],
    [1],
    [1],
    [1],
]

# Device resistances in ohms.
resistances = [
    [345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755,
     257, 646],
    [652, 401, 508, 166, 454, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755,
     257, 646],
    [442, 874, 190, 244, 635, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755,
     257, 646],
    [442, 874, 190, 244, 635, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755,
     257, 646],
    [442, 874, 190, 244, 635, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755, 257, 646, 345, 903, 755,
     257, 646],
]

# Interconnect resistance in ohms.
r_i = 0.5

# Computing the solution.
solution = badcrossbar.compute(applied_voltages, resistances, r_i)

# Printing the current through the word line segment to the left of device in
# the second row and fourth column (Python uses zero-based indexing).
import numpy as np

current = solution
print(solution[0][1])
print(solution.voltages[0])
# plt.imshow(solution.voltages[0], cmap='gray',vmin=0, vmax=1)


R = np.random.randint(100, 2000, (100, 100))
V = np.ones([100, 1], dtype=int)

print(R)
print(V)
solution1 = badcrossbar.compute(V, R, r_i)
plt.imshow(solution1.voltages[0], cmap='gray_r', vmin=0, vmax=1)
plt.show()
plt.imshow(solution1.voltages[1], cmap='gray_r', vmin=0, vmax=1)
plt.show()