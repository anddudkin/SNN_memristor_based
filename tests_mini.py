import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from topology import Connections
from NeuronModels import NeuronLIF

g = torch.tensor([0, 0, 0.5,4,0,7,9,0,0])
print(torch.argmax(g))
conn = Connections(16, 5, "all_to_all")
conn.all_to_all_conn()

conn.initialize_weights("normal")

print(conn.weights)
from visuals import plot_weights, plot_weights_pro

b = conn.weights
c= plot_weights_pro(16,5,conn.weights)
print(plot_weights_pro(16,5,conn.weights))
print(c.shape)