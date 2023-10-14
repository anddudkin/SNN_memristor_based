from random import random

import torch

from anddudkin_mem_project.NeuronModels import NeuronIF
from learning import compute_dw


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

    def inicialize_weights(self):

        """ inicializing random weights"""

        for i in range(len(self.matrix_conn)):
            self.matrix_conn[i][2] = random() / 2

        """ makes matrix of weights [n_in_neurons x n_out_neurons]"""
        self.weights = self.matrix_conn[:, 2].reshape(self.n_in_neurons, self.n_out_neurons)

    def update_w(self, spike_traces_in, spike_traces_out):
        """ take spike traces from Neuron_Model, compute dw and update weights
        \n example:
        out_neurons = Neuron_IF(......)
        update_w(out_neurons.spikes_trace_in,out_neurons.spikes_trace_out)
        """
        spike_traces_out = spike_traces_out.repeat(self.n_in_neurons, 1)
        spike_traces_in = spike_traces_in.reshape(self.n_in_neurons,1).repeat(1, self.n_out_neurons)
        # matrix of dt values
        time_diff = torch.sub(spike_traces_in, spike_traces_out)

        # calling comute_dw function for each dt in matrix
        time_diff.apply_(compute_dw)

        # updating weights

        self.weights = torch.add(self.weights, time_diff)

        self.weights = torch.clamp(self.weights, min=self.w_min, max=self.w_max)


def compute_det_w(matrix_conn, I_in):
    pass
