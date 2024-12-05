import decimal
import math
from numpy import log as ln
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from random import randint

E0 = 3000  # энергия зонда
L = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
L1 = ['M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
name1 = 'Hf'
Eb1 = [2601, 2365, 2108, 1716, 1662, 538, 437, 380, 224, 214, 19, 18]  # энергии связи
Eb1 = {'M1': 2601, 'M2': 2365, 'M3': 2108, 'M4': 1716, 'M5': 1662, 'N1': 538,
       'N2': 437, 'N3': 380, 'N4': 224, 'N5': 214, 'N6': 19, 'N7': 18}
name2 = 'O'
Eb2 = [7709, 926, 794, 779, 101, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0]
with open("gg.txt","w") as g:
    for i in L1[:5]:
        for j in L1[5:]:
            for j1 in L1[5:]:
                if int(j[-1]) <= int(j1[-1]):
                    #print(i + j + j1)
                    z=Eb1[i]-Eb1[j]-Eb1[j1]
                    if 1670 < z < 1690:
                        print(i + j + j1+"="+str(z))

                    g.write('\n'+ i + j + j1+"="+str(Eb1[i])+"-"+str(Eb1[j])
                            +"-"+str(Eb1[j1])+"="+str(z))

n = randint(1, 100) / 100
k = randint(1, 100) / 100
l = randint(1, 100) / 100

plt.vlines(x=516, ymin=0, ymax=1,color=(n, k, l),)
n = randint(1, 100) / 100
k = randint(1, 100) / 100
l = randint(1, 100) / 100
plt.vlines(x=1680, ymin=0, ymax=0.18,color=(n, k, l),)
plt.text(525, 1, "O", size=12)
plt.text(1670, 0.2, "Hf", size=12)
plt.show()