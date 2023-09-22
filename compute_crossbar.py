import time

import torch
import matplotlib.pyplot as plt
import numpy as np


# def weights_inicialization_inferens(G: torch.tensor):

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
    # plt.matshow(I_all.tolist())
    fig.tight_layout()
    # plt.matshow(G.tolist())
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # time.sleep(0.1)
    plt.show()

    return [I_out, G]

def compute_weight_change (U_in):
    pass

