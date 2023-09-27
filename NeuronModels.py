import torch
from topology import construct_matrix_connections
from compute_crossbar import compute_ideal


class Neuron_IF:

    def __init__(self, n_neurons, U_mem, U_tr, U_rest, refr_time, traces=bool):
        """ n_neurons - number of output IF neurons\n
            I_in - Input current\n
            U_tr -  max capacity of neuron (Treshold). If U_mem > U_tr neuron spikes\n
            U_mem - standard membrane potrntial\n
            U_rest - membrane potential while resting (refractory), after neuron spikes\n
            refr_time - refractory period time\n
            """
        self.uniform = None
        self.n_neurons = n_neurons
        self.U_mem = U_mem
        self.U_tr = U_tr
        self.U_rest = U_rest
        self.refr_time = refr_time
        self.traces = traces
        self.spikes_trace = None
        self.U_mem_trace = None
        self.spikes = None
        self.refractor_count = None
        self.U_mem_all_neurons = None

    def initialization(self):
        """ Returns
                    U_mem_all_neurons\n
                    refractor_count\n
                    spikes\n
                    """
        self.U_mem_all_neurons = torch.zeros([self.n_neurons],
                                             dtype=torch.float)
        self.U_mem_all_neurons.fill_(self.U_mem)

        self.refractor_count = torch.zeros([self.n_neurons, 2],
                                           dtype=torch.float)
        self.spikes = torch.zeros([self.n_neurons, 2],
                                  dtype=torch.float)
        for i in range(self.n_neurons):
            self.refractor_count[i][0] = i  # вид [индекс, значение]
            self.spikes[i][0] = i

        if self.traces:
            self.U_mem_trace = []
            self.spikes_trace = []

    def compute_U_mem(self, U_in, conn_matrix):
        """
        Compute U_out for each output neuron
        :param U_in:
        :param conn_matrix:
        :return: tenzor[U_mem] and tenzor [sum of I_out for each output neuron]
        """
        I_for_each_neuron = torch.zeros([self.n_neurons],
                                        dtype=torch.float)
        for idx, i in enumerate(U_in, start=0):
            if self.refractor_count[idx][1] == 0:
                for j in conn_matrix:
                    if idx == j[0]:
                        I_for_each_neuron[idx] += i * j[2]
            else:
                self.refractor_count[idx][1] -= 1
        self.U_mem_all_neurons = torch.add(I_for_each_neuron, self.U_mem_all_neurons)

        return self.U_mem_all_neurons, I_for_each_neuron

    def check_spikes(self):

        for i in range(self.n_neurons):
            self.spikes[i][1] = 0 # обнуляем список импульсов
            if self.U_mem_all_neurons[i] > self.U_tr:  # threshold check

                self.U_mem_all_neurons[i] = self.U_rest  # if spikes rest

                self.spikes[i][1] = 1  # record spike

                self.refractor_count[i][1] = self.refr_time  # start refractor period

                if self.traces:
                    pass  # добавить запись данных для построения графиков
        return self.spikes


def Neuron_LIF(I_in, U_tr, n_neurons: int):
    pass

    # def compute_U_mem(self, I_in):
    #     for idx, i in enumerate(I_in, start=0):
    #         self.U_mem_all_neurons[idx] += i
    #         self.refractor_count[idx][0] += idx
    #         self.spikes[idx][0] = idx  # нумеруем нейроны для отслеживания генерации импульса
    #         if self.U_mem_all_neurons[idx] >= self.U_tr:  # прроверяем привышение мембр. потенциала
    #             self.spikes[idx][1] = 1
    #             self.refractor_count[idx][
    #                 1] = self.refr_time  # если нейрон сгенерировал импульс, начинается рефракторный период
    #     return self.U_mem_all_neurons, self.spikes
