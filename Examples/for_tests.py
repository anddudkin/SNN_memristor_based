import math

import  matplotlib.pyplot as plt
w_m = 0.00005
w_max = 0.01
i_st = 50
def steps(w_m,w_max,steps):
    g = []
    for i in range(steps):
        g.append(w_m+w_max*(1-math.exp(i/steps*math.log(w_m/w_max))))
    return g

plt.plot(list(range(1,i_st+1)),steps(w_m,w_max,i_st), "o")
plt.show()