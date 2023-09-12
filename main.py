import time
jfffll
import torch
import matplotlib.pyplot as plt
import numpy as np

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

    I_out = torch.matmul(G, V_in) #matrix multipl

    for i, j in enumerate(V_in):  # compute currents of each node
        I_all[i] = torch.mul(G[i], int(j))

    print("Currents =  ", I_all)
    print(I_out)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sctr=ax1.matshow(I_all.tolist(),cmap='inferno')
    plt.colorbar(sctr, ax=ax1)
    ax1.set_title('All_Currents')
    gg=ax2.matshow(G.tolist())
    fig.colorbar(gg)
    ax2.set_title('All_weights')
    # plt.matshow(I_all.tolist())
    # plt.matshow(G.tolist())
    plt.show()


    return I_out, G