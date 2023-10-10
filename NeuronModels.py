import torch

from anddudkin_mem_project.learning import compute_dw

from compute_crossbar import compute_ideal


class Neuron_IF:

    def __init__(self, n_neurons, n_neurons_in, U_mem, U_tr, U_rest, refr_time, traces=bool):
        """ n_neurons - number of output IF neurons\n
            I_in - Input current\n
            U_tr -  max capacity of neuron (Treshold). If U_mem > U_tr neuron spikes\n
            U_mem - standard membrane potrntial\n
            U_rest - membrane potential while resting (refractory), after neuron spikes\n
            refr_time - refractory period time\n
            """
        self.dw_all = None
        self.n_neurons_in = n_neurons_in

        self.n_neurons = n_neurons
        self.U_mem = U_mem
        self.U_tr = U_tr
        self.U_rest = U_rest
        self.refr_time = refr_time
        self.traces = traces

        self.U_mem_trace = None
        self.spikes = None
        self.refractor_count = None
        self.U_mem_all_neurons = None

    def initialization(self):
        """
         initialization
                    """
        self.U_mem_all_neurons = torch.zeros([self.n_neurons],
                                             dtype=torch.float)
        self.U_mem_all_neurons.fill_(self.U_mem)

        self.refractor_count = torch.zeros([self.n_neurons, 2],
                                           dtype=torch.float)
        self.spikes = torch.zeros([self.n_neurons, 2],
                                  dtype=torch.float)
        self.time_sim = 0

        for i in range(self.n_neurons):
            self.refractor_count[i][0] = i  # вид [индекс, значение]
            self.spikes[i][0] = i

        if self.traces:
            self.U_mem_trace = torch.zeros([1, self.n_neurons],
                                           dtype=torch.float)
            self.spikes_trace_in = torch.zeros([self.n_neurons_in],
                                               dtype=torch.float)
            self.spikes_trace_out = torch.zeros([self.n_neurons],
                                                dtype=torch.float)

    def compute_U_mem_new(self, U_in, conn_matrix):

        I_for_each_neuron = torch.matmul(U_in, conn_matrix)
        self.time_sim += 1
        for i in range(len(self.U_mem_all_neurons)):
            if self.refractor_count[i][1] == 0:
                self.U_mem_all_neurons[i] += I_for_each_neuron[i]
            else:
                self.refractor_count[i][1] -= 1

        if self.traces:  # spike and U traces
            for i in range(self.n_neurons_in):
                if U_in[i] == 1:
                    self.spikes_trace_in[i] = self.time_sim  # times of spikes
            self.U_mem_trace = torch.cat(                    # stack traces of U_mem for plotting
                (self.U_mem_trace, self.U_mem_all_neurons.reshape(1, len(self.U_mem_all_neurons))), 0)

    def check_spikes(self):
        """
        Checks if neuron spikes
        :return: tenzor [index of neuron, spike (0 or 1)]
        """
        for i in range(self.n_neurons):
            self.spikes[i][1] = 0  # обнуляем список импульсов
            if self.U_mem_all_neurons[i] >= self.U_tr:  # threshold check

                self.U_mem_all_neurons[i] = self.U_rest  # if spikes rest

                self.spikes[i][1] = 1  # record spike

                self.refractor_count[i][1] = self.refr_time  # start refractor period

                if self.traces:
                    for j in range(self.n_neurons):
                        if self.spikes[j][1] == 1:
                            self.spikes_trace_out[j] = self.time_sim  # times of spikes
        return self.spikes

    def update_w_slow(self, conn_matrix):
        self.dw_all = torch.zeros([len(conn_matrix)], dtype=torch.float)

        for indx, i in enumerate(self.spikes_trace_out, start=0):
            for k, j in enumerate(conn_matrix):
                if j[0] == indx:
                    conn_matrix[k][2] += compute_dw(self.spikes_trace_in[int(conn_matrix[k][1])] - i)
                    # self.dw_all[k] = compute_dw(self.spikes_trace_in[int(conn_matrix[k][1])] - i)




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


'''
    def compute_U_mem(self, U_in, conn_matrix):
        """
        Compute U_out for each output neuron
        :param U_in:
        :param conn_matrix:
        :return: tenzor[U_mem] and tenzor [sum of I_out for each output neuron]
        """

        self.time_sim += 1
        I_for_each_neuron = torch.zeros([self.n_neurons],
                                        dtype=torch.float)
        for idx, i in enumerate(U_in, start=0):   # compute dU for each neuron
            for j in conn_matrix:
                if idx == j[1] and self.refractor_count[int(j[0])][1] == 0 and i != 0:
                    I_for_each_neuron[int(j[0])] += i * j[2]
            print(I_for_each_neuron)

        self.U_mem_all_neurons = torch.add(I_for_each_neuron, self.U_mem_all_neurons)  # U + dU

        for i in range(self.n_neurons):   #decrese refractory count
            if self.refractor_count[i][1] > 0:
                self.refractor_count[i][1] -= 1

        if self.traces:  # spike and U traces
            for i in range(self.n_neurons_in):
                if U_in[i] == 1:
                    self.spikes_trace_in[i] = self.time_sim  # times of spikes
            self.U_mem_trace = torch.cat((self.U_mem_trace, self.U_mem_all_neurons.reshape(1,len(self.U_mem_all_neurons))),0)

        return self.U_mem_all_neurons, I_for_each_neuron
'''
