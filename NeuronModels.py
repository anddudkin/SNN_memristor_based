import torch

from compute_crossbar import compute_ideal


class Neuron_IF:

    def __init__(self, n_neurons, U_mem, U_tr, U_rest, refr_time):
        """ n_neurons - number of output IF neurons
            I_in - Input current
            U_tr -  max capacity of neuron (Treshold). If U_mem > U_tr neuron spikes
            U_mem - standard membrane potrntial
            U_rest - membrane potential while resting (refractory), after neuron spikes
            refr_time - refractory period time
            """
        self.n_neurons = n_neurons
        self.U_mem = U_mem
        self.U_tr = U_tr
        self.U_rest = U_rest
        self.refr_time = refr_time

    def initialization(self):
        """ Returns
                    U_mem_all_neurons
                    refractor_count
                    spikes
                    """
        self.U_mem_all_neurons = torch.zeros([self.n_neurons],
                                             dtype=torch.float)
        self.refractor_count = torch.zeros([self.n_neurons,2],
                                           dtype=torch.float)
        self.spikes = torch.zeros([self.n_neurons, 2],
                                  dtype=torch.int8)
        return self.U_mem_all_neurons, self.refractor_count

    def compute_U_mem(self, I_in):
        for idx, i in enumerate(I_in, start=0):
            self.U_mem_all_neurons[idx] += i
            self.refractor_count[idx][0] += idx
            self.spikes[idx][0] = idx  # нумеруем нейроны для отслеживания генерации импульса
            if self.U_mem_all_neurons[idx] >= self.U_tr: #прроверяем привышение мембр. потенциала
                self.spikes[idx][1] = 1
                self.refractor_count[idx][1]= self.refr_time # если нейрон сгенерировал импульс, начинается рефракторный период
        return self.U_mem_all_neurons, self.spikes






def Neuron_LIF(I_in, U_tr, n_neurons: int):
    pass
