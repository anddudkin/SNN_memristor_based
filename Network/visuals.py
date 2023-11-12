import math
import matplotlib.pyplot as plt
import torch

""" Currently under development 
class PlottingMnist:
    def __init__(self, weights):
        pass
"""


def plot_U_mem(n_neurons_out, U_mem_trace):
    x = []
    f, ax = plt.subplots()
    for i in range(len(U_mem_trace)):
        x.append(i)

    for i in range(n_neurons_out):
        ax.plot(x, U_mem_trace[:, i])


def plot_weights_line(n_in, n_out, weights):
    c = []
    for i in range(n_out):
        c.append(weights[:, i].reshape(int(math.sqrt(n_in)), int(math.sqrt(n_in))))

    hh = torch.cat(c, 1)
    return hh


def plot_weights_square(n_in, n_out, weights):
    line = []
    if math.sqrt(n_out) % 2 == 0:
        rows = int(math.sqrt(n_out))
    else:
        rows = int(math.sqrt(n_out) + 1)

    for i in range(n_out):  # list of weighhts for each neuron
        line.append(weights[:, i].reshape(int(math.sqrt(n_in)), int(math.sqrt(n_in))))

    square = []
    if len(line) < rows * rows:
        while len(line) != rows * rows:
            line.append(torch.zeros(14, 14))
    for i in range(rows):
        square.append(torch.cat(line[:rows], 1))
        line = line[rows:]

    square = torch.cat(square, 0)
    return square
