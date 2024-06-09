import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat("excelDataCombined.mat")["data"]

data = np.flip(data, axis=2)
data = np.transpose(data, (1, 2, 0))
print(len((data[2])))
print(data[2][30])
for i in range(20):
    plt.semilogy(data[1][i][:120], data[2][i][:120])
plt.show()
for i in range(20, 53):
    plt.semilogy(data[1][i][:120], data[2][i][:120])
plt.show()
plt.plot(data[1][20], data[0][20])
plt.show()
data = data[:2, :, :]
# print(data[0][1]) #[0] - токи
# print(data[1][1]) #[1] - напруга
# plt.plot(data[1][1],data[0][1])
# plt.show()

for i in range(len(data[0])):
    plt.plot(data[1][i], data[0][i])

plt.show()

print(len(data))
print(len(data[0]))

for i in range(33):
    plt.plot(data[1][i], data[0][i])

plt.show()

for i in range(33, 53):
    plt.plot(data[1][i], data[0][i])

plt.show()

for i in range(33):
    plt.plot(data[1][i][:120], data[0][i][:120])

plt.show()

for i in range(33, 53):
    plt.plot(data[1][i][:120], data[0][i][:120])

plt.show()
