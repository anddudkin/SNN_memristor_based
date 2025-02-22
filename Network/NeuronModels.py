import pickle

import torch
import badcrossbar
from badcrossbar.compute import compute


class NeuronIF:
    """Base class for Integrate and Fire neuron model"""

    def __init__(self, n_neurons_in, n_neurons_out, inh, traces, train, U_mem=0, U_tr=13, U_rest=0, refr_time=5):
        """C

        Args:
            n_neurons_in (int) : number of input IF neurons
            n_neurons_in (int) : number of output IF neurons
            inh (bool)  : activate inhibition or not
            traces (bool)  : activate traces or not
            train (bool) : train or test
            U_tr :  max capacity of neuron (Threshold). If U_mem > U_tr neuron spikes
            U_mem :  initialized membrane potentials
            U_rest  : membrane potential while resting (refractory), after neuron spikes
            refr_time (int) : refractory period time
        """

        self.train = train
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
        self.R_coef = None
        self.Int_coef = None

        self.U_mem_all_neurons = torch.zeros([self.n_neurons_out], dtype=torch.float).fill_(self.U_mem)
        self.U_thresh_all_neurons = torch.zeros([self.n_neurons_out], dtype=torch.float).fill_(self.U_tr)
        self.refractor_count = torch.zeros([self.n_neurons_out], dtype=torch.float)
        self.spikes = torch.zeros([self.n_neurons_out], dtype=torch.int)
        self.time_sim = 0

        # Initializing trace record
        if self.traces:
            self.U_mem_trace = torch.zeros([1, self.n_neurons_out], dtype=torch.float)
            self.spikes_trace_in = torch.zeros([self.n_neurons_in], dtype=torch.float)
            self.spikes_trace_out = torch.zeros([self.n_neurons_out], dtype=torch.float)

        try:
            with open('Res_coeff.pkl', 'rb') as f:
                self.R_coef = pickle.load(f)
            with open('interp_coeff.pkl', 'rb') as f:
                self.Int_coef = pickle.load(f)
        except:
            pass

    def compute_U_mem(self, U_in, weights, k=1, r_line=None, crossbar=False, nonlin=False):
        """Compute I_out for each output neuron and updates U_mem of all neurons

        Args:
            U_in (Tensor): input vector of voltages
            weights(Tensor): matrix of network weights (Connections.weights)

        """
        weights1=weights.clone().detach()
        if not crossbar and not nonlin:
            I_for_each_neuron = torch.matmul(U_in, weights1)

        elif crossbar and not nonlin:
            #####################
            def ff(x):
                return 1/x

            weights1.apply_(ff)
            ##########################
            I_for_each_neuron = torch.squeeze(torch.tensor(
                badcrossbar.compute(U_in.reshape(self.n_neurons_in, 1), weights1, r_line).currents.output))

        else:
            flag = True
            o = 8 * 10 ** (-3)
            cr0 = weights
            while flag:

                solution = badcrossbar.compute(U_in.reshape(self.n_neurons_in, 1), cr0, 1)
                g_g = torch.clone(cr0).detach()
                currents = torch.tensor(solution.currents.device, dtype=torch.float)
                voltage = torch.mul(torch.abs(currents), cr0)

                for i in range(len(cr0)):
                    for j in range(len(cr0[0])):

                        if torch.abs(currents[i][j]) >= 1 * 10 ** (-9) and voltage[i][j] >= 1 * 10 ** -9:
                            g_g[i][j] = voltage[i][j] / (
                                    self.Int_coef[0] * voltage[i][j] ** 3 + self.Int_coef[1] * voltage[i][j] +
                                    self.Int_coef[2] +
                                    self.R_coef[float(weights[i][j])])

                det_g = torch.subtract(g_g, cr0)
                det_g = torch.abs(det_g)
                # print()
                # print(torch.max(det_g))
                # print(torch.max(g_g))
                eps = torch.max(det_g) / (torch.max(cr0))
                print(eps)
                # print(eps)
                cr0 = torch.add(cr0, torch.mul(torch.subtract(g_g, cr0), 1))
                if eps < o:
                    flag = False
                    # print(solution.currents.output)
                    I_for_each_neuron = torch.squeeze(torch.tensor(solution.currents.output))

        self.time_sim += 1

        if self.time_sim > 10000:
            self.time_sim = 0

        for i in range(self.n_neurons_out):
            if self.refractor_count[i] == 0:
                self.U_mem_all_neurons[i] += I_for_each_neuron[i] * k
            else:
                self.refractor_count[i] -= 1

        if self.traces and self.train:  # spike and U traces
            for i in range(self.n_neurons_in):
                if U_in[i] == 1:
                    self.spikes_trace_in[i] = self.time_sim  # times of spikes

    def check_spikes(self):
        """ Checks if neuron spikes, apply inhibition (if needed) and resets U_mem """
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
                            if self.U_mem_all_neurons[j] < self.U_rest:
                                self.U_mem_all_neurons[j] = self.U_rest
                            self.refractor_count[j] = 6
                            self.spikes[j] = 0

                if self.traces and self.train:
                    for j in range(self.n_neurons_out):
                        if self.spikes[j] == 1:
                            self.spikes_trace_out[j] = self.time_sim  # times of spikes
                    # stack traces of U_mem for plotting
                    # self.U_mem_trace = torch.cat(
                    #     (self.U_mem_trace, self.U_mem_all_neurons.reshape(1, len(self.U_mem_all_neurons))), 0)

    def reset_variables(self, U_mem_all, refractor, traces):

        """  Resetting all variables to the original values
         Args:
                U_mem_all (bool) : reset U_mem_all_neurons
                refractor (bool) : reset refractor_count
                traces (bool) : reset traces
        """
        if U_mem_all:
            self.U_mem_all_neurons.fill_(self.U_mem)
        if refractor:
            self.refractor_count = torch.zeros([self.n_neurons_out],
                                               dtype=torch.float).fill_(self.U_mem)
        if traces:
            self.U_mem_trace = torch.zeros([1, self.n_neurons_out], dtype=torch.float)
            self.spikes_trace_in = torch.zeros([self.n_neurons_in], dtype=torch.float)
            self.spikes_trace_out = torch.zeros([self.n_neurons_out], dtype=torch.float)


class NeuronLIF(NeuronIF):
    """ Class for Leaky Integrate and Fire neuron model. Parent class - NeuronIF"""

    def __init__(self, n_neurons_in, n_neurons_out, decay, inh, traces, train, U_mem=0, U_tr=13, U_rest=0,
                 refr_time=5):
        """ NeuronLifAdaptiveThresh

            Args:
                n_neurons_in (int) : number of input LIF neurons
                n_neurons_out (int) : number of output LIF neurons
                decay (float) : leak of membrane potential
                inh (bool)  : activate inhibition or not
                traces (bool)  : activate traces or not
                U_tr  :  max capacity of neuron (Threshold). If U_mem > U_tr neuron spikes
                U_mem :  initialized membrane potentials
                U_rest  : membrane potential while resting (refractory), after neuron spikes
                refr_time (int) : refractory period time
        """
        super().__init__(n_neurons_in, n_neurons_out, inh, traces, train, U_mem, U_tr, U_rest, refr_time)
        self.decay = decay

    def compute_U_mem(self, U_in, weights, k=1, r_line=None, crossbar=False, nonlin=False):
        super().compute_U_mem(U_in, weights, k, r_line, crossbar, nonlin)
        self.U_mem_all_neurons = torch.clamp(self.U_mem_all_neurons, min=self.U_mem)
        self.U_mem_all_neurons = torch.mul(self.U_mem_all_neurons, self.decay)


class NeuronLifAdaptiveThresh(NeuronLIF):
    def __init__(self, n_neurons_in, n_neurons_out, decay, inh, traces, train, U_mem=0, U_tr=13, U_rest=0,
                 refr_time=5):
        """ NeuronLifAdaptiveThresh

            Args:
                n_neurons_in (int) : number of input LIF neurons with adaptive threshold
                n_neurons_out (int) : number of output LIF neurons with adaptive threshold
                decay (float) : leak of membrane potential
                inh (bool)  : activate inhibition or not
                traces (bool)  : activate traces or not
                U_tr :  max capacity of neuron (Threshold). If U_mem > U_tr neuron spikes
                U_mem :  initialized membrane potentials
                U_rest  : membrane potential while resting (refractory), after neuron spikes
                refr_time (int) : refractory period time
        """
        super().__init__(n_neurons_in, n_neurons_out, decay, inh, traces, train, U_mem, U_tr, U_rest, refr_time)

    def check_spikes(self):
        """
        Checks if neuron spikes and reset U_mem

        :return: tensor [index of neuron, spike (0 or 1)]
        """
        # if self.traces and self.train:
        #     self.U_mem_trace = torch.cat(
        #         (self.U_mem_trace,
        #          torch.clamp(self.U_mem_all_neurons.reshape(1, self.n_neurons_out), max=self.U_tr + self.U_tr * 0.2)),
        #         0)
        for i in range(self.n_neurons_out):
            self.spikes[i] = 0  # update spike values
            if self.U_mem_all_neurons[i] >= self.U_thresh_all_neurons[i]:  # threshold check

                if self.train:
                    self.U_thresh_all_neurons[i] += 0.02  # adaptive thresh

                self.U_mem_all_neurons[i] = self.U_rest  # if spike occurs rest

                self.spikes[i] = 1  # record spike

                self.refractor_count[i] = self.refr_time  # start refractor period

                if self.inh:  # inhibition
                    for j in range(self.n_neurons_out):
                        if i != j:
                            self.U_mem_all_neurons[j] -= 5
                            if self.U_mem_all_neurons[j] < self.U_rest:
                                self.U_mem_all_neurons[j] = self.U_rest
                            self.refractor_count[j] = 6
                            self.spikes[j] = 0

                if self.traces and self.train:
                    for j in range(self.n_neurons_out):
                        if self.spikes[j] == 1:
                            self.spikes_trace_out[j] = self.time_sim  # times of spikes

        if self.train:
            self.U_thresh_all_neurons = torch.mul(self.U_thresh_all_neurons, 0.99999)
            self.U_thresh_all_neurons = torch.clamp(self.U_thresh_all_neurons, min=self.U_tr,
                                                    max=self.U_tr * 1.2)

    def check_spikes1(self):
        """
        Checks if neuron spikes and reset U_mem

        :return: tensor [index of neuron, spike (0 or 1)]
        """
        # if self.traces and self.train:
        #     self.U_mem_trace = torch.cat(
        #         (self.U_mem_trace,
        #          torch.clamp(self.U_mem_all_neurons.reshape(1, self.n_neurons_out), max=self.U_tr + self.U_tr * 0.2)),
        #         0)
        for i in range(self.n_neurons_out):
            self.spikes[i] = 0  # update spike values
            if self.U_mem_all_neurons[i] >= self.U_thresh_all_neurons[i]:  # threshold check

                if self.train:
                    self.U_thresh_all_neurons[i] += 0.0002  # adaptive thresh

                self.U_mem_all_neurons[i] = self.U_rest  # if spike occurs rest

                self.spikes[i] = 1  # record spike

                self.refractor_count[i] = self.refr_time  # start refractor period

                if self.inh:  # inhibition
                    for j in range(self.n_neurons_out):
                        if i != j:
                            self.U_mem_all_neurons[j] -= 5/100
                            if self.U_mem_all_neurons[j] < self.U_rest:
                                self.U_mem_all_neurons[j] = self.U_rest
                            self.refractor_count[j] = 6
                            self.spikes[j] = 0

                if self.traces and self.train:
                    for j in range(self.n_neurons_out):
                        if self.spikes[j] == 1:
                            self.spikes_trace_out[j] = self.time_sim  # times of spikes

        if self.train:
            self.U_thresh_all_neurons = torch.mul(self.U_thresh_all_neurons, 0.99999)
            self.U_thresh_all_neurons = torch.clamp(self.U_thresh_all_neurons, min=self.U_tr,
                                                    max=self.U_tr * 1.2)
    def check_spikes3(self):
        """
        Checks if neuron spikes and reset U_mem

        :return: tensor [index of neuron, spike (0 or 1)]
        """
        # if self.traces and self.train:
        #     self.U_mem_trace = torch.cat(
        #         (self.U_mem_trace,
        #          torch.clamp(self.U_mem_all_neurons.reshape(1, self.n_neurons_out), max=self.U_tr + self.U_tr * 0.2)),
        #         0)
        for i in range(self.n_neurons_out):
            self.spikes[i] = 0  # update spike values
            if self.U_mem_all_neurons[i] >= self.U_thresh_all_neurons[i]:  # threshold check

                if self.train:
                    self.U_thresh_all_neurons[i] += 0.000002  # adaptive thresh

                self.U_mem_all_neurons[i] = self.U_rest  # if spike occurs rest

                self.spikes[i] = 1  # record spike

                self.refractor_count[i] = self.refr_time  # start refractor period

                if self.inh:  # inhibition
                    for j in range(self.n_neurons_out):
                        if i != j:
                            self.U_mem_all_neurons[j] -= 5/10000
                            if self.U_mem_all_neurons[j] < self.U_rest:
                                self.U_mem_all_neurons[j] = self.U_rest
                            self.refractor_count[j] = 6
                            self.spikes[j] = 0

                if self.traces and self.train:
                    for j in range(self.n_neurons_out):
                        if self.spikes[j] == 1:
                            self.spikes_trace_out[j] = self.time_sim  # times of spikes

        if self.train:
            self.U_thresh_all_neurons = torch.mul(self.U_thresh_all_neurons, 0.99999)
            self.U_thresh_all_neurons = torch.clamp(self.U_thresh_all_neurons, min=self.U_tr,
                                                    max=self.U_tr * 1.2)
    def save_U_thresh(self, path='thresh.pt'):
        torch.save(self.U_thresh_all_neurons, path)

    def load_U_thresh(self, path='thresh.pt'):
        self.U_thresh_all_neurons = torch.load(path)


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
