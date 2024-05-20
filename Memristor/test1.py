import math
import pickle
import statistics
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

with open('V_0_07.pkl', 'rb') as f:
    voltage = pickle.load(f)
with open('I_0_07.pkl', 'rb') as f:
    current = pickle.load(f)
for i, j in enumerate(voltage):
    # plt.semilogy(j, current[i])
    plt.semilogy(j, current[i])

s, s1 = [], []

for i in range(10):
    s.append(current[i][50])
    s1.append(voltage[i][50])
plt.plot(s1, s, "--", linewidth=3)

print(s)
res_std = statistics.stdev(s)
res_v = statistics.stdev(s1)
print(res_std)
mean_I = []
mean_V = []

for j in range(0, len(voltage[0])):
    tmp = 0
    for i in range(0, len(voltage)):
        tmp = tmp + voltage[i][j]
    mean_V.append(tmp)
for j in range(0, len(current[0])):
    tmp = 0
    for i in range(0, len(current)):
        tmp = tmp + current[i][j]
    mean_I.append(tmp)

for j in range(70):
    mean_V[j] = mean_V[j] / 10.0
    mean_I[j] = mean_I[j] / 10.0
plt.plot(mean_V, mean_I, "--", linewidth=4, color="black")
plt.show()

from scipy.optimize import curve_fit


def mapping1(values_x, a, b, c):
    return a * values_x ** 2 + b * values_x + c


def mapping2(values_x, a, b, c):
    return a * values_x ** 3 + b * values_x + c


def mapping3(values_x, a, b, c):
    return a * values_x ** 3 + b * values_x ** 2 + c


def mapping4(values_x, a, b, c):
    return a * np.exp(b * values_x) + c


mean_I, mean_V = mean_V, mean_I
print(mean_V)
print(mean_I)
v = []

for i in range(1200):
    v.append(i / 10000000)
print(v)
#######
args1, covar = curve_fit(mapping1, mean_V, mean_I)

v1 = []
for i in v:
    v1.append(args1[0] * i ** 2 + args1[1] * i + args1[2])
print(v1)
plt.plot(mean_I, mean_V, "--", linewidth=1, color="black")
plt.plot(v1, v, label="y = a * x^2 + b * x + c")

########
args2, covar = curve_fit(mapping2, mean_V, mean_I)
print("Arguments: ", args2)
v2 = []
for i in v:
    v2.append((args2[0]) * i ** 3 + args2[1] * i + args2[2])
print(v2)
plt.plot(v2, v, label="y = a * x^3 + b * x + c")
#############
args3, covar = curve_fit(mapping3, mean_V, mean_I)

v3 = []
for i in v:
    v3.append(args3[0] * i ** 3 + args3[1] * i ** 2 + args3[2])
print(v3)
plt.plot(v2, v, label="y = a * x^3 + b * x^2 * c")
plt.legend()
plt.show()
###########
# args4, covar = curve_fit(mapping4, mean_V, mean_I)
#
# v4 = []
# for i in v:
#     v4.append(args4[0] * np.exp(args4[1]*i) + args4[2])
# print(v4)
# plt.plot(v, v4, label="..........")
############
ii = []
k=0
for j in range(5):
    ii=[]
    for i in v:
        ii.append((args1[0] * (i) ** 2 + args1[1] * (i) + args1[2])+k)
    print(ii)
    plt.plot(ii, v, label="x")
    k+=2*1.33*10**-6
plt.legend()
plt.show()
