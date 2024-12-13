import pickle
import statistics
import sys
import badcrossbar
import torch
import numpy as np
import matplotlib.pyplot as plt
c = np.random.normal(0, 1, [10,5])
w =np.random.normal(10000, 1000, [10,5])
applied_voltages = np.ones([196, 1])
#solution = badcrossbar.compute(applied_voltages, w, r_i)
print(c)
#print(c1)
x=torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
x=torch.normal(1000, 0.05, size= [1000])
x = np.random.normal(1000, 10, 5000)
print(x)
plt.hist(x,bins=150, density=True)
plt.show()
with open('interp_coeff.pkl', 'rb') as f:
    args2 = pickle.load(f)
print(args2)
v=[]
for i in range(7000):
    v.append(i / 10000)
res_std=1.9546164134240556*10**-6
fig, ax = plt.subplots()
k = 1
k1 = 1
k2 = 1
x = np.array(range(100))
new, new1 = [], []
f=0.9
ff=1
f1,f2=[],[]
for i1 in range(3):
    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []

    # for i in v:
    #     t1.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + (k1 - 1) * res_std)
    for i in v:
        t2.append(args2[0] * (i) ** 3 + args2[1] * (i) + args2[2] + k1 * res_std)

    for i2 in range(20):
        t3 = []
        c = np.random.normal(0, 1, 1)[0]
        print(c)
        for j in v:

            t3.append(args2[0] * (j) ** 3 + args2[1] * (j) + args2[2] +  (k1+c) * res_std)
        if i1==1 and abs(c)<2:
            t3[0] = 0
            color = "black"
            p1 = ax.semilogy(v[1:], t3[1:], linestyle='dashed', linewidth=f, color=color)





    t2[0] = 0

    v[0]=0

    p1=ax.semilogy(v[1:], t2[1:], linestyle='-', linewidth=f+1, color="red",label=str(ff) + " состояние")
    k1 += 2
plt.show()