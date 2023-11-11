import torch
from matplotlib import pyplot as plt
from NeuronModels import NeuronLIF, NeuronIF, NeuronLifAdaptiveThresh
from datasets import MNIST_train_test_14x14, encoding_to_spikes
from topology import Connections


def plot_mnist_test():
    bb = MNIST_train_test_14x14()[0]
    spike_intensity = 1
    b1 = bb[0][0] * spike_intensity
    f = torch.squeeze(encoding_to_spikes(b1, 50))

    g = torch.zeros([14, 14], dtype=torch.float)
    plt.ion()
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)

    for i in f:
        g += i
        ax1.matshow(i, cmap='gray')
        plt.draw()
        plt.pause(0.0001)

    plt.ioff()

    fig1 = plt.figure(figsize=(6, 6))
    ax3 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    g = g / 50
    ax3.set_title('Image Sample')
    ax2.set_title('Recreated image from spike train')
    ax3.imshow(torch.squeeze(bb[0][0]), cmap='gray', vmin=0, vmax=1)
    ax2.imshow(g, cmap='gray', vmin=0, vmax=1)
    plt.show()

plot_mnist_test()
def if_neuron_test():
    """ Visualize how single IF neuron works.
    """
    lif_test = NeuronIF(4, 2, U_tr=100, U_rest=0, refr_time=5, traces=True, inh=False, train=True)
    conn = Connections(4, 2, "all_to_all")
    conn.all_to_all_conn()
    conn.initialize_weights("normal")

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
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'U_mem', fontsize=12)
    plt.plot(x, y)

    U2 = torch.tensor([0.8, 0.5, 0.99, 0.7], dtype=torch.float)
    y, x = [], []
    lif_test.reset_variables(True, True, True)

    for i in range(1000):
        lif_test.compute_U_mem(torch.bernoulli(U2), conn.weights)
        lif_test.check_spikes()
        y.append(float(lif_test.U_mem_all_neurons[0]))
        x.append(i)

    plt.figure(2)
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'U_mem', fontsize=12)
    plt.plot(x, y)
    plt.show()


def lif_neuron_test():
    """ Visualize how single LIF neuron works. decay corresponds to the amount of leakage of membrane potential
    https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html
    """
    lif_test = NeuronLIF(4, 2, decay=0.95, U_tr=41, U_rest=0, refr_time=5, traces=True, inh=False, train=True)
    conn = Connections(4, 2, "all_to_all")
    conn.all_to_all_conn()
    conn.initialize_weights("normal")

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
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'U_mem', fontsize=12)
    plt.plot(x, y)

    U2 = torch.tensor([0.8, 0.5, 0.99, 0.7], dtype=torch.float)
    y, x = [], []
    lif_test.reset_variables(True, True, True)

    for i in range(1000):
        lif_test.compute_U_mem(torch.bernoulli(U2), conn.weights)
        lif_test.check_spikes()
        y.append(float(lif_test.U_mem_all_neurons[0]))
        x.append(i)

    plt.figure(2)
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'U_mem', fontsize=12)
    plt.plot(x, y)
    plt.show()


def lif_thresh_neuron_test():
    """ Visualize how single LIF with adaptive neuron works."""
    lif_test = NeuronLifAdaptiveThresh(4, 2, decay=0.97, U_tr=14, U_rest=0, refr_time=5, traces=True, inh=False, train=True)
    conn = Connections(4, 2, "all_to_all")
    conn.all_to_all_conn()
    conn.initialize_weights("normal")

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
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'U_mem', fontsize=12)
    plt.plot(x, y)

    U2 = torch.tensor([0.9, 0.9, 0.99, 0.9], dtype=torch.float)
    y, x = [], []
    lif_test.reset_variables(True, True, True)

    for i in range(1000):
        lif_test.compute_U_mem(torch.bernoulli(U2), conn.weights)
        lif_test.check_spikes()
        y.append(float(lif_test.U_mem_all_neurons[0]))
        x.append(i)

    plt.figure(2)
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'U_mem', fontsize=12)
    plt.plot(x, y)
    plt.axhline(y=lif_test.U_tr, color='k')
    plt.show()

