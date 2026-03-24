import pickle

from matplotlib import pyplot as plt

with open('snn_weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)

print(loaded_weights)
plt.imshow(loaded_weights)
plt.show()