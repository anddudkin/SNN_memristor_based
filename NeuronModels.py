import torch

from learning import compute_dw

from compute_crossbar import compute_ideal


class NeuronIF:
    """Base class for Integrate and Fire neuron model"""

    def __init__(self, n_neurons_in, n_neurons_out, inh=bool, traces=bool, U_mem=0, U_tr=13, U_rest=0, refr_time=5):
        """C

        Args:
            n_neurons (int) : number of output IF neurons
            U_tr :  max capacity of neuron (Threshold). If U_mem > U_tr neuron spikes
            U_mem :  initialized membrane potentials
            U_rest  : membrane potential while resting (refractory), after neuron spikes
            refr_time (int) : refractory period time
        """

        self.inh = inh
        self.n_neurons_in = n_neurons_in
        self.n_neurons_out = n_neurons_out
        self.U_mem = U_mem
        self.U_tr = U_tr
        self.U_rest = U_rest
        self.refr_time = refr_time
        self.traces = traces

        self.dw_all = None
        self.U_mem_trace = None
        # Initializing values
        self.U_mem_all_neurons = torch.zeros([self.n_neurons_out],
                                             dtype=torch.float).fill_(self.U_mem)
        self.U_thresh_all_neurons = torch.zeros([self.n_neurons_out],
                                                dtype=torch.float).fill_(self.U_tr)
        # self.U_mem_all_neurons.fill_(self.U_mem)

        self.refractor_count = torch.zeros([self.n_neurons_out],
                                           dtype=torch.float)
        self.spikes = torch.zeros([self.n_neurons_out],
                                  dtype=torch.float)
        self.time_sim = 0

        # Initializing trace record
        if self.traces:
            self.U_mem_trace = torch.zeros([1, self.n_neurons_out],
                                           dtype=torch.float)
            self.spikes_trace_in = torch.zeros([self.n_neurons_in],
                                               dtype=torch.float)
            self.spikes_trace_out = torch.zeros([self.n_neurons_out],
                                                dtype=torch.float)

    def compute_U_mem(self, U_in, weights):
        """Compute I_out for each output neuron and updates U_mem of all neurons

        Args:
            U_in (Tensor): input vector of voltages
            weights(Tensor): matrix of network weights (Connections.weights)
        """
        I_for_each_neuron = torch.matmul(U_in, weights)
        self.time_sim += 1
        for i in range(self.n_neurons_out):
            if self.refractor_count[i] == 0:
                self.U_mem_all_neurons[i] += I_for_each_neuron[i]
            else:
                self.refractor_count[i] -= 1

        if self.traces:  # spike and U traces
            for i in range(self.n_neurons_in):
                if U_in[i] == 1:
                    self.spikes_trace_in[i] = self.time_sim  # times of spikes
            # stack traces of U_mem for plotting

            self.U_mem_trace = torch.cat(
                (self.U_mem_trace, self.U_mem_all_neurons.reshape(1, len(self.U_mem_all_neurons))), 0)

    def check_spikes(self):
        """
        Checks if neuron spikes and reset U_mem

        :return: tensor [index of neuron, spike (0 or 1)]
        """
        for i in range(self.n_neurons_out):
            self.spikes[i] = 0  # update spike values
            if self.U_mem_all_neurons[i] >= self.U_tr:  # threshold check

                self.U_mem_all_neurons[i] = self.U_rest  # if spike occurs rest

                self.spikes[i] = 1  # record spike

                self.refractor_count[i] = self.refr_time  # start refractor period

                if self.inh:  # inhibition
                    for j in range(self.n_neurons_out):
                        if i != j:
                            self.U_mem_all_neurons[j] -= 10
                            self.refractor_count[j] = 6
                            self.spikes[j] = 0

                if self.traces:
                    for j in range(self.n_neurons_out):
                        if self.spikes[j] == 1:
                            self.spikes_trace_out[j] = self.time_sim  # times of spikes

    def reset_variables(self):

        """  Resetting all variables to the original values """

        self.U_mem_all_neurons = torch.zeros([self.n_neurons_out],
                                             dtype=torch.float).fill_(self.U_mem)
        self.U_mem_all_neurons.fill_(self.U_mem)

        self.refractor_count = torch.zeros([self.n_neurons_out],
                                           dtype=torch.float).fill_(self.U_mem)


class NeuronLIF(NeuronIF):
    """ Class for Leaky Integrate and Fire neuron model. Parent class - NeuronIF"""

    def __init__(self, n_neurons_in, n_neurons_out, decay, inh=bool, traces=bool, U_mem=0, U_tr=13, U_rest=0, refr_time=5):
        super().__init__(n_neurons_in, n_neurons_out, inh, traces, U_mem, U_tr, U_rest, refr_time)
        self.decay = decay

    def compute_U_mem(self, U_in, weights):
        super().compute_U_mem(U_in, weights)

        self.U_mem_all_neurons = torch.clamp(self.U_mem_all_neurons, min=self.U_mem)
        self.U_mem_all_neurons = torch.mul(self.U_mem_all_neurons, self.decay)


class NeuronLifAdaptiveThresh(NeuronLIF):
    def __init__(self, n_neurons_in, n_neurons_out, decay, inh=bool, traces=bool, U_mem=0, U_tr=13, U_rest=0, refr_time=5):
        """ NeuronLifAdaptiveThresh

            Args:
                n_neurons_in (int) : number of input IF neurons
                n_neurons_out (int) : number of output IF neurons
                U_tr :  max capacity of neuron (Threshold). If U_mem > U_tr neuron spikes
                U_mem :  initialized membrane potentials
                U_rest  : membrane potential while resting (refractory), after neuron spikes
                refr_time (int) : refractory period time
        """
        super().__init__(n_neurons_in, n_neurons_out, decay, inh, traces, U_mem, U_tr, U_rest, refr_time)

    def check_spikes(self):
        """
        Checks if neuron spikes and reset U_mem

        :return: tensor [index of neuron, spike (0 or 1)]
        """
        for i in range(self.n_neurons_out):
            self.spikes[i] = 0  # update spike values
            if self.U_mem_all_neurons[i] >= self.U_thresh_all_neurons[i]:  # threshold check

                self.U_thresh_all_neurons[i] += 0.02
                self.U_mem_all_neurons[i] = self.U_rest  # if spike occurs rest

                self.spikes[i] = 1  # record spike

                self.refractor_count[i] = self.refr_time  # start refractor period

                if self.inh:  # inhibition
                    for j in range(self.n_neurons_out):
                        if i != j:
                            self.U_mem_all_neurons[j] -= 5
                            self.refractor_count[j] = 6
                            self.spikes[j] = 0

                if self.traces:
                    for j in range(self.n_neurons_out):
                        if self.spikes[j] == 1:
                            self.spikes_trace_out[j] = self.time_sim  # times of spikes

        self.U_thresh_all_neurons = torch.mul(self.U_thresh_all_neurons, 0.99999)
        self.U_thresh_all_neurons = torch.clamp(self.U_thresh_all_neurons, min=self.U_tr, max=self.U_tr + self.U_tr * 0.2)


class NeuronInhibitory:
    def __init__(self, n_neurons, inh):
        self.n_neurons = n_neurons
        self.inh = inh

    def compute_inhibition(self, spikes, U_mem_all_neurons):
        for i in range(self.n_neurons):
            if spikes[i] == 1:
                for j in range(self.n_neurons):
                    if j != i:
                        U_mem_all_neurons[j] -= self.inh

        return U_mem_all_neurons


'''
    def compute_U_mem_slow(self, U_in, conn_matrix):
        """
        Compute U_out for each output neuron
        :param U_in:
        :param conn_matrix:
        :return: tenzor[U_mem] and tenzor [sum of I_out for each output neuron]
        """

        self.time_sim += 1
        I_for_each_neuron = torch.zeros([self.n_neurons_out],
                                        dtype=torch.float)
        for idx, i in enumerate(U_in, start=0):   # compute dU for each neuron
            for j in conn_matrix:
                if idx == j[1] and self.refractor_count[int(j[0])][1] == 0 and i != 0:
                    I_for_each_neuron[int(j[0])] += i * j[2]
            print(I_for_each_neuron)

        self.U_mem_all_neurons = torch.add(I_for_each_neuron, self.U_mem_all_neurons)  # U + dU

        for i in range(self.n_neurons_out):   #decrese refractory count
            if self.refractor_count[i][1] > 0:
                self.refractor_count[i][1] -= 1

        if self.traces:  # spike and U traces
            for i in range(self.n_neurons_in):
                if U_in[i] == 1:
                    self.spikes_trace_in[i] = self.time_sim  # times of spikes
            self.U_mem_trace = torch.cat((self.U_mem_trace, self.U_mem_all_neurons.reshape(1,len(self.U_mem_all_neurons))),0)

        return self.U_mem_all_neurons, I_for_each_neuron
'''

'''
    def initialization(self):
        """ Initializing values

        Returns:
            Initialized values
        """

        self.U_mem_all_neurons = torch.zeros([self.n_neurons_out],
                                             dtype=torch.float)
        self.U_mem_all_neurons.fill_(self.U_mem)

        self.refractor_count = torch.zeros([self.n_neurons_out, 2],
                                           dtype=torch.float)
        self.spikes = torch.zeros([self.n_neurons_out, 2],
                                  dtype=torch.float)
        self.time_sim = 0

        for i in range(self.n_neurons_out):
            self.refractor_count[i][0] = i  # вид [индекс, значение]
            self.spikes[i][0] = i

        if self.traces:
            self.U_mem_trace = torch.zeros([1, self.n_neurons_out],
                                           dtype=torch.float)
            self.spikes_trace_in = torch.zeros([self.n_neurons_in],
                                               dtype=torch.float)
            self.spikes_trace_out = torch.zeros([self.n_neurons_out],
                                                dtype=torch.float)
'''
