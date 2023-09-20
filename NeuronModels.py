import torch

from compute_crossbar import compute_ideal


def Neuron_IF(I_in, n_neurons: int, U_tr, U_rest=10, refr=5):
    """I_in - Input current
        U_tr -  max capacity of neuron (Treshold). If U_mem > U_tr neuron spikes
    """

    if I_in.size != n_neurons:
        print("Size of input currents must be the same as number of neurons")
        print("I_in.size= ", I_in.size, "neurons= ", n_neurons)

    U_all_neurons = torch.zeros([n_neurons],
                                dtype=torch.float)  # array with U_mem = 0 for all neurons #каждый раз при вызове мембранный потенциал сбрасывется!!!
    spikes = torch.zeros([n_neurons, 2], dtype=torch.float)

    for idx, i in enumerate(I_in, start=0):
        U_all_neurons[idx] += i
        spikes[idx][0] = idx  # нумеруем нейроны для отслеживания генерации импульса
        if U_all_neurons[idx] >= U_tr:
            spikes[idx][1] = 1
    return [U_all_neurons, spikes]


def Neuron_LIF(I_in, U_tr, n_neurons: int):
    pass
