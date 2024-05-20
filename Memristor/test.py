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
    s.append(current[i][20])
    s1.append(voltage[i][20])
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
print(mean_I)
print(mean_V)

for j in range(70):
    mean_V[j] = mean_V[j] / 10.0
    mean_I[j] = mean_I[j] / 10.0
plt.plot(mean_V, mean_I, "--", linewidth=4, color="black")
plt.show()
print(list(mean_V))
from scipy.optimize import curve_fit


def mapping1(values_x, a, b, c):
    return a * values_x ** 2 + b * values_x + c


def mapping2(values_x, a, b, c):
    return a * values_x ** 3 + b * values_x + c


def mapping3(values_x, a, b, c):
    return a * values_x ** 3 + b * values_x ** 2 + c


def mapping4(values_x, a, b, c):
    return a * np.exp(b * values_x) + c


v = []

for i in range(700):
    v.append(i / 1000)
print(v)
#######
args1, covar = curve_fit(mapping1, mean_V, mean_I)

v1 = []
for i in v:
    v1.append(args1[0] * i ** 2 + args1[1] * i + args1[2])
print(v1)
plt.plot(mean_V, mean_I, "--", linewidth=1, color="black")
plt.plot(v, v1, label="y = a * x^2 + b * x + c")
########
args2, covar = curve_fit(mapping2, mean_V, mean_I)
print("Arguments: ", args2)
v2 = []
for i in v:
    v2.append((args2[0]) * i ** 3 + args2[1] * i + args2[2])
print(v2)
plt.plot(v, v2, label="y = a * x^3 + b * x + c")
#############
args3, covar = curve_fit(mapping3, mean_V, mean_I)

v3 = []
for i in v:
    v3.append(args3[0] * i ** 3 + args3[1] * i ** 2 + args3[2])
print(v3)
plt.plot(v, v3, label="y = a * x^3 + b * x^2 * c")
plt.legend()
plt.show()
k = 0
for i in range(200):
    v4 = []
    for i in v:
        v4.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2]+k)
    k += 2*res_std
    plt.semilogy(v, v4)

plt.show()
t1,t2,t3=[],[],[]
for i in v:
    t1.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2]+res_std)
for i in v:
    t2.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2]+10*res_std)
for i in v:
    t3.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2]+20*res_std)
for i in range(len(v)):
    print(t2[i]-t1[i])
fig, ax = plt.subplots()
ax.plot(v, t2, '-')
ax.fill_between(v, t1, t3, alpha=0.2)
plt.show()

###########.
# args4, covar = curve_fit(mapping4, mean_V, mean_I)
#
# v4 = []
# for i in v:
#     v4.append(args4[0] * np.exp(args4[1]*i) + args4[2])
# print(v4)
# plt.plot(v, v4, label="..........")
############
# k = 0
# for j in range(50):
#     ii = []
#     for i in v:
#         ii.append(i/((args2[0]+k) * (i) ** 3 + args2[1] * (i) + args2[2]))
#
#     plt.semilogy(v, ii, label="x")
#     k += res_std
#
# plt.show()
