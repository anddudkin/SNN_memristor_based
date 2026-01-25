import numpy as np
from get_data import read_data_full
X, y = read_data_full()
X = np.array(X)
y = np.array(y)
import matplotlib.pyplot as plt
import torch.nn.functional as F
X = X.astype('float32') / -255
#X = np.expand_dims(X, axis=-1)  # форма (N, 28, 28, 1)
import torch
import kornia.filters as KF
print(X.shape)
print(X[0].shape)
print(X[0])
el = 5500
plt.subplot(1, 5, 1)
plt.imshow(X[el],cmap='binary')
plt.title(f'Class {y[el]}, 28x28')

torch_tensor = torch.from_numpy(X[el])
print(torch_tensor.shape)
output_tensor = F.interpolate(torch_tensor.unsqueeze(0).unsqueeze(1), size=(14,14), mode='nearest')
plt.subplot(1, 5, 2)

plt.imshow(output_tensor.squeeze(),cmap='binary')
plt.title(f'14x14')
kernel = torch.tensor([
    [-1.,-1., -1.],
    [-1.,  9., -1.],
    [-1., -1., -1.]
])/10
kernel = kernel.unsqueeze(0).unsqueeze(0)
new_t=F.conv2d(output_tensor, kernel, padding=0)

plt.subplot(1, 5, 3)
plt.imshow(new_t.squeeze(), cmap='binary')
plt.title(f'conv2d')

# Plot in the second position

plt.subplot(1, 5, 4)
edges = KF.sobel(output_tensor)
plt.imshow(edges.squeeze(),cmap='binary')
plt.title(f'sobel')


plt.subplot(1, 5, 5)
edges1 = KF.canny(output_tensor)
plt.imshow(edges1[0].squeeze(),cmap='binary')
plt.title(f'canny')
plt.tight_layout()
plt.show()

print(y)