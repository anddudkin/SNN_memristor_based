from random import random
import torch
from learning import compute_dw

"""
https://arxiv.org/ftp/arxiv/papers/1710/1710.04734.pdf
"""


class Connections:
    def __init__(self, n_in_neurons, n_out_neurons, type_conn="all_to_all", w_min=0, w_max=1, decay=(False, 0.9999)):
        """ Type of connection: 1) "all_to_all" 2)....."""
        self.decay = decay
        self.w_max = w_max
        self.w_min = w_min
        self.weights = None
        self.w = None
        self.matrix_con_weights = None
        self.matrix_conn = None
        self.type = type_conn
        self.n_in_neurons = n_in_neurons
        self.n_out_neurons = n_out_neurons

    def all_to_all_conn(self):
        """Construct matrix with id of IN/OUT neurons and weights [out,in,w]"""

        self.matrix_conn = torch.zeros([self.n_in_neurons * self.n_out_neurons, 3], dtype=torch.float)

        if self.type == "all_to_all":
            while self.matrix_conn[-1][0] == 0:
                ind = 0
                for i in range(self.n_in_neurons):
                    for j in range(self.n_out_neurons):
                        self.matrix_conn[ind][0] = i  # [in,out,w]
                        self.matrix_conn[ind][1] = j
                        ind += 1

    def initialize_weights(self, dis="normal"):

        """ initializing random weights

        Args:
            dis (str) : type of weights distribution

        """
        if dis == "rand":
            for i in range(len(self.matrix_conn)):
                self.matrix_conn[i][2] = random()

                self.weights = self.matrix_conn[:, 2].reshape(self.n_in_neurons,
                                                              self.n_out_neurons)  # makes matrix of weights [n_in_neurons x n_out_neurons]
        elif dis == "normal":
            self.weights = self.matrix_conn[:, 2].reshape(self.n_in_neurons, self.n_out_neurons)
            self.weights = self.weights.normal_(mean=0.7, std=0.2)

    def update_w(self, spike_traces_in, spike_traces_out, spikes):

        """ Take spike traces from NeuronModels, compute dw and update weights )

        Args:
            spike_traces_in : traces of input spikes_
            spike_traces_out : traces of output spikes_
            spikes: spikes of neurons each time step

        \n example:
        out_neurons = Neuron_IF(......)
        update_w(out_neurons.spikes_trace_in,out_neurons.spikes_trace_out

        """

        spike_traces_out = spike_traces_out.repeat(self.n_in_neurons, 1)
        spike_traces_in = spike_traces_in.reshape(self.n_in_neurons, 1).repeat(1, self.n_out_neurons)

        time_diff = torch.sub(spike_traces_in, spike_traces_out)  # matrix of dt values

        for i, sp in enumerate(spikes, start=0):  # updating weights (only weights of neuron that spiked)
            if sp == 1:
                time_diff[:, i].apply_(compute_dw)  # calling compute_dw function for each dt in matrix

                self.weights[:, i] = torch.add(self.weights[:, i], time_diff[:, i])

        if self.decay[0]:
            self.weights = torch.mul(self.weights, self.decay[1])  # weight decay

        self.weights = torch.clamp(self.weights, min=self.w_min, max=self.w_max)
