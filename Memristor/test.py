import math
import pickle
import statistics

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
    return a * exp(b * values_x) + c
v=[]

for i in range(700):
    v.append(i/1000)
print(v)
#######
args, covar = curve_fit(mapping1, mean_V, mean_I)

ii=[]
for i in v:
    ii.append(args[0] * i ** 2 + args[1] * i + args[2])
print(ii)
plt.plot(mean_V, mean_I, "--", linewidth=1, color="black")
plt.plot(v,ii, label="y = a * x^2 + b * x + c")
########
args1, covar = curve_fit(mapping2, mean_V, mean_I)
print("Arguments: ", args)
ii=[]
for i in v:
    ii.append((args[0]) * i ** 3 + args[1] * i + args[2])
print(ii)
plt.plot(v,ii,label="y = a * x^3 + b * x + c")
#############
args1, covar = curve_fit(mapping3, mean_V, mean_I)

ii=[]
for i in v:

    ii.append(args[0] * (i ** 3) + args[1] * i + args[2])
print(ii)
plt.plot(v,ii,label="y = a * x^3 + b * x^2 * c")
###########




args, covar = curve_fit(mapping2, mean_I, mean_V)
print("Arguments: ", args)
ii=[]
for i in v:
    ii.append((args1[0]) * (i) ** 3 + args1[1] * (i) + args1[2])
print(ii)
plt.plot(v,ii,label="x")
plt.legend()
plt.show()