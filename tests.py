from random import random as rand

import torch
from main import compute_ideal
import time





#hh = torch.randperm(20)
a = torch.tensor([
    [0.1, 0.1, 0.1],
    [0.1, 0.2, 0.1],
    [0.2, 0.6, 0.7],
])

b = torch.tensor(
    [10., 1., 5.,1.]
)

for i in range(100):
    x = int(torch.randint(1000,[1]))
    y = int(torch.randint(10,[1]))
    h = torch.randint(x, (20, 20))
    hh = torch.randint(y, ([20]))
    compute_ideal(hh,h)

# print(a.shape[1])
# print(b.__len__())
# print(torch.split(a,1))
# print(h)
# print(hh)