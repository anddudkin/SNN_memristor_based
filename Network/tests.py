import pickle
import matplotlib.pyplot as plt
import torch
import badcrossbar

from Network.visuals import plot_weights_square

w = torch.load("C:/Users/anddu/Desktop/7сем/2_Работа/SNN-memristor-based/test/4 класаа/weights_tensor.pt")
plt.imshow(plot_weights_square(196, 50, w), cmap='YlOrBr', vmin=0.00004, vmax=0.001, interpolation='None')
plt.show()