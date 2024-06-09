import math

import torch
import matplotlib.pyplot as plt
import badcrossbar


class TransformToCrossbarBase:
    def __init__(self, weights, R_min=5000, R_max=25000, r_line=1):
        self.U_drop = None
        self.V_drop = None
        self.I_out = None
        self.r_line = r_line
        self.R_max = R_max
        self.R_min = R_min
        self.weights = weights
        self.weights_Om = None
        self.n_neurons_in = len(weights)
        self.n_neurons_out = len(weights[0])
        self.G_max = 1 / self.R_min
        self.G_min = 1 / self.R_max

        self.w_min = torch.min(self.weights)
        self.w_max = torch.max(self.weights)

        if 1 / self.w_max < self.R_min:
            self.k = self.R_min / self.w_max
        else:
            self.k = 1

        def GtoR(x):
            if x <= self.G_min * 10000:
                return self.R_max
            else:
                return 1 / (x / self.k)

        self.weights_Siemens = torch.clone(self.weights).detach()  # in siemens
        self.weights_Om = self.weights.apply_(GtoR)  # in Oms

    def compute_crossbar(self, U_in):
        solution = badcrossbar.compute(U_in.reshape(self.n_neurons_in, 1), self.weights, r_i=self.r_line)
        self.I_out = solution.currents.output
        self.U_drop = solution.voltages.word_line

    def transform_with_experemental_data(self, data_R):
        def f(x):  # перерасчет с коэффициентов в диапазон эксперементальных сопротивлений
            return x / (self.R_max / max(data_R))

        self.weights_Om = self.weights_Om.apply_(f)
        self.R_max = torch.max(self.weights_Om)
        self.R_min = torch.min(self.weights_Om)

        data_R.sort()

        def nearest_value(value):
            '''Поиск ближайшего значения до value в списке items'''
            found = data_R[0]  # найденное значение (первоначально первое)
            for item in data_R:
                if value <= 1000:
                    return data_R[0]
                if abs(item - value) < abs(found - value):
                    found = item
            return found

        self.weights_Om = self.weights_Om.apply_(nearest_value)
        self.R_max = torch.max(self.weights_Om)
        self.R_min = torch.min(self.weights_Om)
        self.weights_Siemens = None
        self.G_max = None
        self.G_min = None
        self.w_min = None
        self.w_max = None

    def compute_crossbar_nonlinear(self, U_in, ):
        def rtog(x):
            return 1 / float(x)

        def gtor(x):
            return 1 / float(x)

        o = 10 ** (-6)

        print("iterar")
        cr0 = torch.clone(self.weights_Om)
        crG = torch.clone(self.weights)
        flag = True

        while flag:

            if cr0[0][0] > 1:
                g_g = torch.clone(cr0)
            else:
                g_g = torch.clone(cr0.apply_(gtor))

            solution = badcrossbar.compute(U_in, g_g, 1)
            voltage = solution.voltages.word_line
            # currents = solution.currents.device

            for i in range(len(cr0)):
                for j in range(len(cr0[0])):
                    cr0[i][j] = 2 * 1 / crG[i][j] * 0.1 * math.exp(5 * math.sqrt(voltage[i][j] / 4))

            det_g = torch.subtract(cr0, g_g)

            det_g = torch.abs(det_g)

            eps = torch.max(det_g) / (torch.max(g_g))

            print(eps)

            if eps < o:
                flag = False
                print(solution.voltages.word_line)
                print(solution.currents.device)

            return solution.currents.device

    def plot_crossbar_U(self, U_in):
        j = plt.imshow(self.U_drop, cmap='gray_r', vmin=torch.min(U_in), vmax=torch.max(U_in), interpolation='None')
        plt.colorbar(j, fraction=0.12, pad=0.04)
        plt.show()

    def plot_crossbar_weights(self):
        j1 = plt.imshow(self.weights, cmap='gray', vmin=self.R_min, vmax=self.R_max,
                        interpolation='None')
        plt.colorbar(j1, fraction=0.12, pad=0.04)
        plt.show()


# def weights_inicialization_inferens(G: torch.tensor):
class CrossbarLearn:
    def __init__(self, n_neurons_in, n_neurons_out, R_min, R_max, r_line):
        self.r_line = r_line
        self.weights = None
        self.n_neurons_in = n_neurons_in
        self.n_neurons_out = n_neurons_out
        self.R_min = R_min
        self.R_max = R_max
        self.G_min = 1 / self.R_max
        self.G_max = 1 / self.R_min

    def init_weights(self):
        self.weights = torch.tensor((self.n_neurons_in, self.n_neurons_out), dtype=torch.float)

    def compute_I_out(self, U_in):
        solution = badcrossbar.compute(U_in.reshape(self.n_neurons_in, 1), self.weights, r_i=self.r_line)


def compute_ideal(V_in: torch.tensor, G: torch.tensor):
    '''  Compute ideal crossbar
            V_in - vector of input viltages
            G - matrix for conductances of memristors
            shape  G = m x n  V = n
    Output: I_out - vector of output currents
            I_all - Matrix of currents of each memristor
    '''
    if G.shape[1] != V_in.__len__():
        print('INCORRECT SHAPE////input shape needed G = m x n , V = n')
        exit()
    I_all = torch.empty(G.shape)

    I_out = torch.matmul(G, V_in)  # matrix multipl

    for i, j in enumerate(V_in):  # compute currents of each node
        I_all[i] = torch.mul(G[i], j)

    print("Currents_all =  ", I_all)
    print("I_out= ", I_out)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    p1 = ax1.matshow(I_all.tolist(), cmap='inferno')
    plt.colorbar(p1, fraction=0.046, pad=0.04)
    ax1.set_title('All_Currents')

    p2 = ax2.matshow(G.tolist())
    fig.colorbar(p2, fraction=0.046, pad=0.04)
    ax2.set_title('All_weights')

    fig.tight_layout()
    plt.show()

    return [I_out, G]


def compute_weight_change(U_in):
    pass
