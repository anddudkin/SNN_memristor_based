import torch
from matplotlib import pyplot as plt
from anddudkin_mem_project.NeuronModels import NeuronLIF
from anddudkin_mem_project.topology import Connections


def lif_neuron_test():
    """ Visualize how single LIF neuron works. decay corresponds to the amount of leakage of membrane potential
    https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html
    """
    lif_test = NeuronLIF(4, 2, decay=0.97, U_tr=100, U_rest=-20, refr_time=5, traces=True, U_mem_min=0)
    conn = Connections(4, 2, "all_to_all")
    conn.all_to_all_conn()
    conn.inicialize_weights()

    U = torch.tensor([1, 1, 1, 1], dtype=torch.float)
    U1 = torch.tensor([0, 0, 0, 0], dtype=torch.float)

    y, x = [], []

    for i in range(200):
        lif_test.compute_U_mem(U, conn.weights)
        y.append(float(lif_test.U_mem_all_neurons[0]))
        x.append(i)
    for i in range(200, 400, 1):
        lif_test.compute_U_mem(U1, conn.weights)
        y.append(float(lif_test.U_mem_all_neurons[0]))
        x.append(i)

    plt.figure(1)
    plt.plot(x, y)

    U2 = torch.tensor([0.8, 0.9, 0.9, 0.9], dtype=torch.float)
    y, x = [], []
    lif_test.reset_variables()

    for i in range(1000):
        lif_test.compute_U_mem(torch.bernoulli(U2), conn.weights)
        y.append(float(lif_test.U_mem_all_neurons[0]))
        x.append(i)

    plt.figure(2)
    plt.plot(x, y)
    plt.show()
