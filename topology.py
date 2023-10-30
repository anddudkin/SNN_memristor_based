from random import random

import torch

from NeuronModels import NeuronIF
from learning import compute_dw

"""
https://arxiv.org/ftp/arxiv/papers/1710/1710.04734.pdf
"""


class Connections:
    def __init__(self, n_in_neurons, n_out_neurons, type_conn="all_to_all", w_min=0, w_max=1):
        """ type of connection: 1) "all_to_all" 2).....
                """
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
        """construct matrix with id of IN/OUT neurons and weights [out,in,w]"""

        self.matrix_conn = torch.zeros([self.n_in_neurons * self.n_out_neurons, 3], dtype=torch.float)

        if self.type == "all_to_all":
            while self.matrix_conn[self.n_in_neurons * self.n_out_neurons - 1][0] == 0:
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
            self.weights = self.weights.normal_(mean=0.2, std=0.1)

        elif dis == "xavier":
            self.weights = self.matrix_conn[:, 2].reshape(self.n_in_neurons, self.n_out_neurons)
            self.weights = torch.nn.init.xavier_uniform_(self.weights, gain=1.)

    def update_w(self, spike_traces_in, spike_traces_out, spikes):

        """ Take spike traces from Neuron_Model, compute dw and update weights )

        Args:
            spike_traces_in :
            spike_traces_out :

        \n example:
        out_neurons = Neuron_IF(......)
        update_w(out_neurons.spikes_trace_in,out_neurons.spikes_trace_out
        """

        spike_traces_out = spike_traces_out.repeat(self.n_in_neurons, 1)
        spike_traces_in = spike_traces_in.reshape(self.n_in_neurons, 1).repeat(1, self.n_out_neurons)
        # matrix of dt values
        """
        time_diff = torch.zeros([self.n_in_neurons, self.n_out_neurons])

        for i, sp in enumerate(spikes, start=0):
            if sp == 1:
                time_diff[:, i] = torch.sub(spike_traces_in[:, i], spike_traces_out[:, i])
                print(time_diff)
                time_diff[:, i].apply_(compute_dw)"""

        time_diff = torch.sub(spike_traces_in, spike_traces_out)
        # calling compute_dw function for each dt in matrix
        time_diff.apply_(compute_dw)
        torch.set_printoptions(threshold=10_000)

        # updating weights (only weights of neuron that spiked)
        for i, sp in enumerate(spikes, start=0):
            if sp == 1:
                self.weights[:, i] = torch.add(self.weights[:, i], time_diff[:, i])

        self.weights = torch.mul(self.weights, 0.99985)  # weight decay

        self.weights = torch.clamp(self.weights, min=self.w_min, max=self.w_max)


def compute_det_w(matrix_conn, I_in):
    pass
