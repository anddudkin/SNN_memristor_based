import math
from multiprocessing import Pool
from random import random

import numpy as np
import torch
from Network.learning import compute_dw, compute_dw1

"""
https://arxiv.org/ftp/arxiv/papers/1710/1710.04734.pdf
"""


class Connections:
    """Class constructs connections (synapses) and operates with their wheights"""

    def __init__(self, n_in_neurons, n_out_neurons, type_conn="all_to_all", w_min=0, w_max=1, decay=(False, 0.9999)):

        """ Type of connection: 1) "all_to_all" 2).....

            Args:
                n_in_neurons (int) : number of input neurons
                n_out_neurons (int) : number of output neurons
                type_conn (str)  : connection type
                w_min (float)  : minimum weights value
                w_max (float) : maximum weights value
                decay (tuple)  : weights decay (True/False, decay value)
        """

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

        """ Initializing random weights

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
            self.weights = self.weights.normal_(mean=self.w_max * 0.7, std=self.w_max * 0.2)
            self.weights = torch.clamp(self.weights, min=self.w_min, max=self.w_max)

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

    def update_w1(self, spike_traces_in, spike_traces_out, spikes):  # модификация для реальных значений проводимости

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
                time_diff[:, i].apply_(compute_dw1)  # calling compute_dw function for each dt in matrix

                self.weights[:, i] = torch.add(self.weights[:, i], time_diff[:, i])

        if self.decay[0]:
            self.weights = torch.mul(self.weights, self.decay[1])  # weight decay

        self.weights = torch.clamp(self.weights, min=self.w_min, max=self.w_max)

    def update_w2(self, spike_traces_in, spike_traces_out, spikes, d_min, d_max,
                  number_states, descrete_st = (False,None),
                  nonlinear=False):  # модификация для реальных значений проводимости и линейного дискретного диапазона состояний

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

        def find_nearest(array, value):
            idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
            return val

        def bisection(array, value):
            '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
            and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
            to indicate that ``value`` is out of range below and above respectively.'''
            n = len(array)
            if value < array[0]:
                return array[0]
            elif value > array[n - 1]:
                return array[-1]
            jl = 0  # Initialize lower
            ju = n - 1  # and upper limits.
            while ju - jl > 1:  # If we are not yet done,
                jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
                if value >= array[jm]:
                    jl = jm  # and replace either the lower limit
                else:
                    ju = jm  # or the upper limit, as appropriate.
                # Repeat until the test condition is satisfied.
            if value == array[0]:  # edge cases at bottom
                return array[0]
            elif (value - array[jl]) <= (array[jl + 1] - value):
                return array[jl]
            else:
                return array[jl + 1]

        def linespace_diff_dens(start, end, num1, num2, s):
            # s - разделения числовой прямой на части 0ю5 - пополам
            num1 += 1
            l_all = end - start
            first_point = l_all * s
            return np.concatenate((np.linspace(start, first_point, num1)[:-1], np.linspace(first_point, end, num2)))

        def steps(w_m, w_max, steps):
            g = []
            for i1 in range(steps):
                g.append(w_m + w_max * (1 - math.exp(i1 / steps * math.log(w_m / w_max))))
            return g

        if not nonlinear:
            discrete_states = np.linspace(d_min, d_max, number_states)
        elif nonlinear and not descrete_st[0]:
            if number_states == 256:
                num1 = 100
                num2=  156
            elif number_states == 128:
                num1 = 28
                num2 = 100
            elif number_states == 64:
                num1 = 26
                num2 = 38
            elif number_states == 32:
                num1 = 13
                num2 = 19
            elif number_states == 16:
                num1 = 7
                num2 = 9
            discrete_states = linespace_diff_dens(0.00005, 0.01, num1, num2, 0.75)
        elif nonlinear and descrete_st[0]:
            discrete_states= descrete_st[1]


        for i, sp in enumerate(spikes, start=0):  # updating weights (only weights of neuron that spiked)
            if sp == 1:
                time_diff[:, i].apply_(compute_dw1)  # calling compute_dw function for each dt in matrix

                self.weights[:, i] = torch.add(self.weights[:, i], time_diff[:, i])
                # print(self.weights)
                for j in range(len(self.weights[:, i])):  # приводим к ближайшему дискретному состоянию
                    self.weights[:, i][j] = bisection(discrete_states, self.weights[:, i][j])
                # print(self.weights)

        if self.decay[0]:
            self.weights = torch.mul(self.weights, self.decay[1])  # weight decay

        self.weights = torch.clamp(self.weights, min=self.w_min, max=self.w_max)

    def save_weights(self, path='weights_tensor.pt'):
        torch.save(self.weights, path)

    def load_weights(self, path='weights_tensor.pt'):
        self.weights = torch.load(path)
