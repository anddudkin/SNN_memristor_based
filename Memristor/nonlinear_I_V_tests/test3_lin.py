import math
import pickle
import statistics
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import exp

import badcrossbar
from Memristor import compute_crossbar
from compute_crossbar import TransformToCrossbarBase
import torch


def gtor(x):
    return 1 / float(x)


w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
#w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")

d = {}

with open('../Res_states.pkl', 'rb') as f:
    r = pickle.load(f)
# with open('Volt_Amper.pkl', 'rb') as f:
#     U_I = pickle.load(f)
with open('../Res_coeff.pkl', 'rb') as f:
    R_coef = pickle.load(f)
with open('../interp_coeff.pkl', 'rb') as f:
    Int_coef = pickle.load(f)
c = TransformToCrossbarBase(w, 5000, 25000, 0)
print(R_coef)
print(Int_coef)

# c.plot_crossbar_weights()

c.transform_with_experemental_data(r)
print(c.weights_Om)
# c.plot_crossbar_weights()

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
#input_spikes = encoding_to_spikes(data_train[0][0], 2)
with open("../spikes.pt", 'rb') as f:
    input_spikes= pickle.load(f)
V = np.ones([196, 1]) / 2
crR = c.weights_Om

k = 0
eps = 0
print("iterations stars......")
cr0 = crR.clone().detach()
flag = True
sol = None
g_iter = None
gr_v, gr_i, gr_g = [], [], []

torch.set_printoptions(threshold=10_000)
V1 = input_spikes[0].reshape(196, 1) / 2

torch.set_printoptions(precision=6)


solution = badcrossbar.compute(V1, cr0, 0)

currents = torch.tensor(solution.currents.device, dtype=torch.float)

f1 = plt.imshow(solution.currents.device, cmap='gray_r', vmin=-0.000085, vmax=0.00025, interpolation='None')
plt.colorbar(f1, fraction=0.12, pad=0.04)
plt.show()
f2 = plt.imshow(solution.currents.device, cmap="gnuplot2" , vmin=-0.000085,vmax=0.00025, interpolation='None')
plt.colorbar(f2, fraction=0.12, pad=0.04)
plt.show()
f3 = plt.imshow(torch.abs(torch.tensor(solution.currents.device,dtype=torch.float)), vmin=-0.000085,vmax=0.00025,interpolation='None')
plt.colorbar(f3, fraction=0.12, pad=0.04)
plt.show()






