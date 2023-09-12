import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
print(MNIST)
datasets.MNIST(root='./data', train=False, download=True, transform=None)

transform = transforms.Compose([
    transforms.ToTensor()])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)
print(dataset1[0])
print(dataset2)


plt.imshow(dataset1.data[0])
plt.title(dataset1[0][1])
plt.show()
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1)
test_loader = torch.utils.data.DataLoader(dataset2)
print(train_loader)
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(data, target)
