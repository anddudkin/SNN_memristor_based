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

with open('Res_states.pkl', 'rb') as f:
    r = pickle.load(f)
with open('Volt_Amper.pkl', 'rb') as f:
    U_I = pickle.load(f)
c = TransformToCrossbarBase(w, 5000, 25000, 0)

# c.plot_crossbar_weights()
import  torch
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
o = 4 * 10 ** (-3)
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
#print(V1)
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
n=49
while flag:
    # if cr0[0][0] > 1:
    #     g_g = cr0.clone().detach()
    # else:
    #     g_g = (cr0.apply_(gtor)).clone().detach()
    g_g = cr0.clone().detach()
    g_g = torch.add(g_g, torch.subtract(cr0,g_g) * 10)
    solution = badcrossbar.compute(V1, g_g, 1)

    # voltage = torch.subtract(torch.tensor(solution.voltages.word_line, dtype=torch.float),
    #                          torch.tensor(solution.voltages.bit_line, dtype=torch.float))
    currents = torch.tensor(solution.currents.device, dtype=torch.float)
    #currents=torch.abs(currents)
    voltage=torch.tensor(solution.voltages.word_line, dtype=torch.float)
    # f1 = plt.imshow(currents, cmap='gray_r', interpolation='None')
    # plt.colorbar(f1, fraction=0.12, pad=0.04)
    # plt.show()
    # f2 = plt.imshow(voltage, cmap='gray_r', interpolation='None')
    # plt.colorbar(f2, fraction=0.12, pad=0.04)
    # plt.show()
    ####

    gr_v.append(abs(currents[n][n]*crR[n][n]))
    gr_i.append(abs(currents[n][n]))
    for i in range(len(cr0)):
        for j in range(len(cr0[0])):

            #cr0[i][j] = voltage[i][j] / (U_I[round(float(crR[i][j]), 0)][round(float(voltage[i][j]), 5)])
            #cr0[i][j] = torch.abs(voltage[i][j]) / (U_I[round(float(crR[i][j]), 0)][round(float(torch.abs(voltage[i][j])), 5)])
            if torch.abs(currents[i][j]) > 2*10**(-5) :
                #cr0[i][j] = (cr0[i][j]*torch.abs(currents[i][j]))/ (U_I[round(float(crR[i][j]), 0)][round(float(torch.abs(cr0[i][j]*currents[i][j])), 5)])
                cr0[i][j] = (cr0[i][j] * torch.abs(currents[i][j])) / (U_I[round(float(crR[i][j]), 0)][round(float(torch.abs(g_g[i][j]*currents[i][j])), 5)])
            
            # print(cr0[i][j])

    gr_g.append(cr0[n][n])
    det_g = torch.subtract(cr0, g_g)

    det_g = torch.abs(det_g)
    print()
    print(torch.mean(det_g))
    print(torch.mean(g_g))
    eps = torch.max(det_g) / (torch.max(g_g))

    print(eps)

    if eps < o:
        flag = False
        # print(solution.voltages.word_line)
        sol = solution
        g_iter = cr0
        # print(solution.currents.device)
        # print(g_iter)
        f1 = plt.imshow(solution.currents.device, cmap='gray_r', interpolation='None', vmax=0.00015)
        plt.colorbar(f1, fraction=0.12, pad=0.04)
        plt.show()
        f2 = plt.imshow(voltage, cmap='gray_r', vmin=0, vmax=0.5, interpolation='None')
        plt.colorbar(f2, fraction=0.12, pad=0.04)
        plt.show()
        print(solution.currents.output)
        voltage_nonlin = torch.tensor(voltage, dtype=torch.float)
        currents_nonlin = torch.tensor(solution.currents.device, dtype=torch.float)
        print(solution.voltages.bit_line)

plt.semilogy(U_I[round(float(crR[n][n]), 0)].keys(), U_I[round(float(crR[n][n]), 0)].values())
plt.semilogy(U_I[round(float(crR[n][n]), 0)].keys(), U_I[round(float(crR[n][n]), 0)].values())
plt.semilogy(gr_v, gr_i, "--")
n = []
for i in range(len(gr_v)):
    n.append(i)
for i, txt in enumerate(n):
    plt.annotate(txt, (gr_v[i], gr_i[i]), fontsize=12)
plt.show()
"""
det_u = torch.subtract(voltage_lin, voltage_nonlin)
# det_u = torch.abs(det_u)
det_i = torch.subtract(currents_lin, currents_nonlin)
# det_i = torch.abs(det_i)
f3 = plt.imshow(det_u, cmap='gray_r', interpolation='None')
plt.colorbar(f3, fraction=0.12, pad=0.04)
plt.show()
f4 = plt.imshow(det_i, cmap='gray_r', interpolation='None')
plt.colorbar(f4, fraction=0.12, pad=0.04)
plt.show()
"""
