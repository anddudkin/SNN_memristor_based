import decimal
import math
from numpy import log as ln
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from random import randint

o = [954.6, 1462.6, 1479.6]
hf = [948.6, 1049.6, 1106.6, 1262.6, 1272.6, 1467.6, 1468.6]
n = randint(1, 100) / 100
k = randint(1, 100) / 100
l = randint(1, 100) / 100
for i in o:
    plt.vlines(x=i, ymin=0, ymax=1, color=(n, k, l), linestyles="--")
n = randint(1, 100) / 100
k = randint(1, 100) / 100
l = randint(1, 100) / 100
for i in hf:
    plt.vlines(x=i, ymin=0, ymax=1, color=(n, k, l), linestyles="-")
plt.show()
name=['K', 'L1', 'L2', 'L3']
name1=['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
o1 = [1, 0.02, 0.02]
hf1 = [0.685622955, 0.382212186, 0.269500719, 0.071898691, 0.064140947, 0.000160699, 0.00015793]
o1e = [954.6, 1462.6, 1479.6]
hf1e = [948.6, 1049.6, 1106.6, 1262.6, 1272.6, 1467.6, 1468.6]
for j, i in enumerate(o1):
    plt.vlines(x=o1e[j], ymin=0, ymax=i, color="red", linestyles="--")
    plt.text(o1e[j], i+0.02, "O"+name[j], size=8)
n = randint(1, 100) / 100
k = randint(1, 100) / 100
l = randint(1, 100) / 100
for j, i in enumerate(hf1):
    plt.vlines(x=hf1e[j],  ymin=0, ymax=i, color="black", linestyles="-")
    plt.text(hf1e[j], i+0.02,"Hf"+name1[j], size=8)
plt.show()
