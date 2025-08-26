import math
import pickle
import statistics
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


#w = torch.load("G:/Другие компьютеры/Ноутбук/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")
w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/50_3000/tau 4/weights_tensor.pt")

d = {}

with open('../Res_states.pkl', 'rb') as f:
    r = pickle.load(f)
with open('Volt_Amper.pkl', 'rb') as f:
    U_I = pickle.load(f)

c = TransformToCrossbarBase(w, 5000, 25000, 0)

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
# for i in range(len(V1)):
#     if V1[i][0] == 0:
#         V1[i][0] = 0.
# print(V1)
# for i in U_I:
#     plt.semilogy(U_I[i].keys(),U_I[i].values())
#     #print(U_I[i].keys())
# plt.show()
# """
# solution = badcrossbar.compute(input_spikes[0].reshape(196, 1)/2, cr0, 1)
#
#
# voltage_lin = torch.subtract(torch.tensor(solution.voltages.word_line, dtype=torch.float),
#                              torch.tensor(solution.voltages.bit_line, dtype=torch.float))
# currents_lin = torch.tensor(solution.currents.device, dtype=torch.float)
#
# f1=plt.imshow(solution.currents.device,cmap='gray_r', interpolation='None',vmax=0.00015)
# plt.colorbar(f1, fraction=0.12, pad=0.04)
# plt.show()
# f2=plt.imshow(voltage_lin,cmap='gray_r', vmin=0, vmax=0.5, interpolation='None')
# plt.colorbar(f2, fraction=0.12, pad=0.04)
# plt.show()
# """
n = 1
o = 1* 10 ** (-2)
torch.set_printoptions(precision=6)
while flag:

    solution = badcrossbar.compute(V1, cr0, 1)
    g_g = torch.clone(cr0).detach()
    # voltage = torch.subtract(torch.tensor(solution.voltages.word_line, dtype=torch.float),
    #                            torch.tensor(solution.voltages.bit_line, dtype=torch.float))
    currents = torch.tensor(solution.currents.device, dtype=torch.float)

    # currents=torch.abs(currents)
    # voltage = torch.tensor(solution.voltages.word_line, dtype=torch.float)
    # f1 = plt.imshow(currents, cmap='gray_r', interpolation='None')
    # plt.colorbar(f1, fraction=0.12, pad=0.04)
    # plt.show()
    # f2 = plt.imshow(voltage, cmap='gray_r', interpolation='None')
    # plt.colorbar(f2, fraction=0.12, pad=0.04)
    # plt.show()
    ####
    # print(cr0)
    # print(torch.abs(currents))
    voltage = torch.mul(torch.abs(currents), cr0)
    # print(voltage)
    # print()
    gr_v.append(voltage[n][n])
    gr_i.append(abs(currents[n][n]))
    #print(currents)
    for i in range(len(cr0)):
        for j in range(len(cr0[0])):

            if torch.abs(currents[i][j]) >= 1 * 10 ** (-6) and voltage[i][j] >= 1*10**-6:
                g_g[i][j] = voltage[i][j] / (U_I[round(float(crR[i][j]), 0)][round(float(voltage[i][j]), 6)])
            if voltage[i][j] < 1*10**-6:
                    #print(round(float(voltage[i][j]), 6))
                 pass
                #print(U_I[round(float(crR[i][j]), 0)][round(float(voltage[i][j]), 6)], voltage[i][j], g_g[i][j])
    # torch.nan_to_num(g_g, nan=1.0, posinf=1.0)
    gr_g.append(cr0[n][n])
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
        f1 = plt.imshow(solution.currents.device, cmap='gray_r', interpolation='None')
        plt.colorbar(f1, fraction=0.12, pad=0.04)
        plt.show()
        f1 = plt.imshow(solution.currents.device, cmap="gnuplot2", interpolation='None')
        plt.colorbar(f1, fraction=0.12, pad=0.04)
        plt.show()
        f1 = plt.imshow(torch.abs(torch.tensor(solution.currents.device, dtype=torch.float)), interpolation='None')
        plt.colorbar(f1, fraction=0.12, pad=0.04)
        plt.show()
        print(solution.currents.output)

plt.semilogy(U_I[round(float(crR[n][n]), 0)].keys(), U_I[round(float(crR[n][n]), 0)].values(), label="Нелинейная ВАХ")
plt.semilogy(gr_v, gr_i, "--", label="Последовательное приближение")
n = []
for i in range(len(gr_v)):
    n.append(i)
for i, txt in enumerate(n):
    if i<6:
        plt.annotate(txt, (gr_v[i], gr_i[i]), fontsize=12)
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.tick_params(which='minor', direction="in")
plt.xlabel("Напряжение, В")
plt.ylabel("Ток, А")
plt.legend()
plt.show()
