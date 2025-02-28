import math

import  matplotlib.pyplot as plt
w_m = 0.00005
w_max = 0.01
i_st = 128

figure, axis = plt.subplots(2, 2,figsize=(10, 10))
def steps(w_m,w_max,steps):
    g = []
    for i in range(steps):
        g.append(w_m+w_max*(1-math.exp(i/steps*math.log(w_m/w_max))))
    return g

g1=steps(w_m,w_max,i_st)
for i in range(127):
    print(i, g1[i+1] - g1[i], g1[i+1],g1[i])

def steps1(w_m,w_max,steps):
    g = []
    for i1 in range(steps):
        g.append(w_m*math.exp(i1/steps*math.log(w_max/w_m)))
    return g

axis[0, 0].plot(list(range(1,i_st+1)),steps(w_m,w_max,i_st), "o",markersize=0.5)
axis[0, 0].set_title("up")
d1= []
for i in range(127):
    d1.append(g1[i+1] - g1[i])
    print(i, g1[i+1] - g1[i], g1[i+1],g1[i])
# For Cosine Function
axis[0, 1].plot(list(range(1,i_st)),d1,"o",markersize=0.5)
axis[0, 1].set_title("dx")


g1=steps1(w_m,w_max,i_st)
d1=[]
for i in range(127):
    d1.append(g1[i+1] - g1[i])
    print(i, g1[i+1] - g1[i], g1[i+1],g1[i])

axis[1, 0].plot(list(range(1,i_st+1)),steps1(w_m,w_max,i_st), "o",markersize=0.5)
axis[1, 0].set_title("dwn")

axis[1, 1].plot(list(range(1,i_st)),d1,"o",markersize=0.5)
axis[1, 1].set_title("dx")

plt.show()
from Network.learning import plot_classic_STDP, plot_simple_stdp
plot_simple_stdp()