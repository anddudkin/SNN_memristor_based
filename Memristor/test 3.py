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

with open('Res_states.pkl', 'rb') as f:
    r = pickle.load(f)
# with open('Volt_Amper.pkl', 'rb') as f:
#     U_I = pickle.load(f)
with open('Res_coeff.pkl', 'rb') as f:
    R_coef = pickle.load(f)
with open('interp_coeff.pkl', 'rb') as f:
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
input_spikes = encoding_to_spikes(data_train[0][0], 2)
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

n1 = 172
n2=16
print("spectate memrisor ", n1, n2, "Res=", cr0[n1][n2])
o = 5* 10 ** (-3)
torch.set_printoptions(precision=6)
while flag:

    solution = badcrossbar.compute(V1, cr0, 1)
    g_g = torch.clone(cr0).detach()
    currents = torch.tensor(solution.currents.device, dtype=torch.float)
    voltage = torch.mul(torch.abs(currents), cr0)

    gr_v.append(voltage[n1][n2])
    gr_i.append(abs(currents[n1][n2]))
    # f2 = plt.imshow(voltage, cmap='gray_r', vmin=0, vmax=0.5, interpolation='None')
    # plt.colorbar(f2, fraction=0.12, pad=0.04)
    # plt.show()
    for i in range(len(cr0)):
        for j in range(len(cr0[0])):

            if torch.abs(currents[i][j]) >= 1 * 10 ** (-9) and voltage[i][j] >= 1*10**-9:
                g_g[i][j] = voltage[i][j] / (Int_coef[0] * voltage[i][j] ** 3 + Int_coef[1] * voltage[i][j] + Int_coef[2]+R_coef[float(crR[i][j])])

    # torch.nan_to_num(g_g, nan=1.0, posinf=1.0)
    gr_g.append(cr0[n1][n2])
    # print(g_g)
    det_g = torch.subtract(g_g, cr0)

    det_g = torch.abs(det_g)
    # torch.nan_to_num(det_g, nan=1.0, posinf=1.0)
    print()
    print(torch.max(det_g))
    print(torch.max(g_g))
    eps = torch.max(det_g) / (torch.max(cr0))
    print(eps)

    cr0 = torch.add(cr0, torch.mul(torch.subtract(g_g, cr0), 1))

    if eps < o:
        flag = False
        sol = solution
        g_iter = cr0
        print(solution.currents.output)

v=[]
for i in range(70000):
    v.append(i / 100000)
ii=[]
for i in v:
    ii.append(Int_coef[0] * i ** 3 + Int_coef[1] * i + Int_coef[2]+R_coef[float(crR[n1][n2])])
plt.semilogy(v,ii)
plt.semilogy(gr_v, gr_i, "--")
n = []
for i in range(len(gr_v)):
    n.append(i)
for i, txt in enumerate(n):
    plt.annotate(txt, (gr_v[i], gr_i[i]), fontsize=12)
plt.show()
