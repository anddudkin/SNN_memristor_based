import numpy as np
from get_data import read_data_full

import matplotlib.pyplot as plt
import torch.nn.functional as F

#X = np.expand_dims(X, axis=-1)  # форма (N, 28, 28, 1)
import torch
import kornia.filters as KF

from PIL import Image
import torchvision.transforms as transforms




image_path = "2.JPG"
image = Image.open(image_path).convert('L')  # 'L' - режим оттенков серого

# Преобразование в тензор
transform = transforms.ToTensor()
output_tensor= transform(image)






plt.imshow(output_tensor.squeeze(),cmap='binary')
plt.show()
kernel = torch.tensor([
    [-1.,-1., -1.],
    [-1.,  9., -1.],
    [-1., -1., -1.]
])/10
kernel = kernel.unsqueeze(0).unsqueeze(0)
new_t=F.conv2d(output_tensor, kernel, padding=0)


plt.imshow(new_t.squeeze(), cmap='binary')
plt.title(f'conv2d')
plt.show()
# Plot in the second position

# plt.subplot(1, 5, 4)
# edges = KF.sobel(output_tensor)
# plt.imshow(edges.squeeze(),cmap='binary')
# plt.title(f'sobel')



edges1 = KF.canny(output_tensor.unsqueeze(1))
plt.imshow(edges1[0].squeeze(),cmap='binary')
plt.title(f'canny')
plt.tight_layout()
plt.show()

print(y)