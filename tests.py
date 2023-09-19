from random import random as rand

import torch
from torchvision.datasets import MNIST

from compute_crossbar import compute_ideal
import time
from datasets import MNIST_train_test
# print(MNIST_train_test())




#hh = torch.randperm(20)
a = torch.tensor([
    [0.1, 0.1, 0.1],
    [0.1, 0.2, 0.1],
    [0.2, 0.6, 0.7],
])

b = torch.tensor(
    [10., 1., 5.,1.]
)

x = int(torch.randint(1000, [1]))
y = int(torch.randint(1, 10, [1]))
h = torch.randint(x, (40, 40))
hh = torch.randint(y, ([40]))
# compute_ideal(hh, h)
b=torch.rand((10))
print(b[1])
print(type(b[1]))
#
#
# print(a.shape[1])
# print(b.__len__())
# print(torch.split(a,1))
# print(h)
# print(hh)