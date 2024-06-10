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

for i in range(7000):
    v.append(i / 10000)

#######
args1, covar = curve_fit(mapping1, mean_V, mean_I)

v1 = []
for i in v:
    v1.append(args1[0] * i ** 2 + args1[1] * i + args1[2])

plt.plot(mean_V, mean_I, "--", linewidth=1, color="black")
plt.plot(v, v1, label="y = a * x^2 + b * x + c")
########
args2, covar = curve_fit(mapping2, mean_V, mean_I)
print("Arguments: ", args2)
v2 = []
for i in v:
    v2.append((args2[0]) * i ** 3 + args2[1] * i + args2[2])

plt.plot(v, v2, label="y = a * x^3 + b * x + c")
#############
args3, covar = curve_fit(mapping3, mean_V, mean_I)

v3 = []
for i in v:
    v3.append(args3[0] * i ** 3 + args3[1] * i ** 2 + args3[2])

plt.plot(v, v3, label="y = a * x^3 + b * x^2 * c")
plt.legend()
plt.show()
k = 0
for i in range(80):
    v4 = []
    for i in v:
        v4.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + k)
    k += 2 * res_std
    v4[0] = 0
    plt.semilogy(v, v4, '-', linewidth=0.5)

# plt.show()


fig, ax = plt.subplots()
k = 1
k1 = 1
k2 = 1
x = np.array(range(100))
new, new1 = [], []

for i in range(20):
    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []

    for i in v:
        t1.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + (k1 - 1) * res_std)
    for i in v:
        t2.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + k1 * res_std)
    new.append(t2)
    for i in v:
        t3.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + (k1 + 1) * res_std)

    t1[0] = 0
    t2[0] = 0
    t3[0] = 0
    r, g, b = None, None, None
    for ggg in range(11):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        y = ggg + np.random.rand(100) * 0.25
    # ax.semilogy(v, t4, '--', color="black", linewidth=0.5)
    ax.semilogy(v, t2, '--', color="black", linewidth=0.5)
    ax.fill_between(v, t2, t1, alpha=0.1, color=[r, g, b])
    ax.fill_between(v, t3, t2, alpha=0.1, color=[r, g, b])

    k1 += 2

plt.show()
# new=new1+new
new_r = []
for i in new:
    h = []
    for j in range(len(v)):
        if i[j] == 0:
            h.append(0)
        else:
            h.append(v[j] / i[j])
    new_r.append(h)
print(v[2000])
print(v[4000])
print(len(v[2000:3000]))
print(len(new_r))
for i in new_r:
    print(i[20])
R_list = []

for i in new_r:
    R_list.append(round(sum(i[2000:2500]) / 500, 0))

print(R_list)
with open("Res_states.pkl", 'wb') as f:
    pickle.dump(R_list, f)

d_r = {}
k1 = 0
for k in range(20):
    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []

    d_u = {}

    for i in v:
        t2.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + k1 * res_std)
        d_u[i] = args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + k1 * res_std
    t2[0] = 0
    new.append(t2)
    plt.semilogy(v, t2)
    d_u[0]=0
    d_r[R_list[k]] = d_u
    k1 += 2
plt.show()
"""словари {R : {U : I, U:I, U:I}, R : {U : I, U:I, U:I} }"""
with open("Volt_Amper.pkl", 'wb') as f:
    pickle.dump(d_r, f)
# print(d_r)
