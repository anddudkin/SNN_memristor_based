import math
import pickle
import statistics
import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import exp

import badcrossbar


def gtor(x):
    return 1 / float(x)

w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
#w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
# j1 = plt.imshow(w, cmap='gray_r', vmin=0, vmax=1,
#                 interpolation='None')
# plt.colorbar(j1, fraction=0.12, pad=0.04)
# plt.show()
d = {}
from Memristor import compute_crossbar
from compute_crossbar import TransformToCrossbarBase

with open('../Res_states.pkl', 'rb') as f:
    r = pickle.load(f)
with open('Volt_Amper.pkl', 'rb') as f:
    U_I = pickle.load(f)
c = TransformToCrossbarBase(w, 5000, 25000, 0)

# c.plot_crossbar_weights()

c.transform_with_experemental_data(r)
print(c.weights_Om)
# c.plot_crossbar_weights()
print(r)
g = []
for i in range(len(c.weights_Om)):
    for j in range(len(c.weights_Om[0])):
        if c.weights_Om[i][j] not in g:
            g.append(int(c.weights_Om[i][j]))

print("эксперимент: ", r)
print("в кроссбаре: ", sorted(g))
print("использовано состояний", len(g) / len(r) * 100, " ->", len(g), "/", len(r))

from Network.datasets import encoding_to_spikes, MNIST_train_test_14x14

data_train = MNIST_train_test_14x14()[0]
input_spikes = encoding_to_spikes(data_train[0][0], 2)
V = np.ones([196, 1]) / 2
crR = c.weights_Om

V1 = np.array(input_spikes[0].reshape(196, 1) / 2)

solution = badcrossbar.compute(V1, crR, 1)
print(solution)