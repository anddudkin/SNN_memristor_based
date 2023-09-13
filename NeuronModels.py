import torch

from compute import compute_ideal



def Neuron_Integrator(I_in, U_tr, n_neurons: int):
    '''I_in - Input current
        U_tr -  max capacity of neuron (Treshold). If U_mem > U_tr neuron spikes
     '''

    if I_in.size != n_neurons:
        print("Size of input currents must be the same as numer of neurons")
        print("I_in.size= ", I_in.size, "neurons= ", n_neurons)
    U_all_neurons = torch.zeros([n_neurons],
                                dtype=torch.long)  # array with U_mem = 0 for all neurons #каждый раз при вызове мембранный потенциал сбрасывется!!!
    spikes = torch.zeros([n_neurons, 2], dtype=torch.long)
    for idx, i in enumerate(I_in,start=0):
        U_all_neurons[idx] += i
        spikes[idx][0] = idx  # нумеруем нейроны для отслеживания генерации импульса
        if U_all_neurons[idx] >= U_tr:
            spikes[idx][1] = 1
    return [U_all_neurons, spikes]



