import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def rand_in_U(n_in_neurons):  # [1., 1., 0., 0., 1., 1., 1., 0., 1., 0.]
    U = torch.bernoulli(torch.randn(n_in_neurons).uniform_())
    return U


def rand_values():
    G = torch.rand((40, 40))
    U = torch.rand(40)
    return [U, G]


def MNIST_train_test():
    datasets.MNIST(root='./data', train=True, download=True, transform=None)

    transform = transforms.Compose([
        transforms.ToTensor()])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    return dataset1, dataset2


import os


<<<<<<< HEAD
def MNIST_train_test_14x14(root="C:/Users/anddu/Documents/GitHub/anddudkin_mem_project/data"):

=======
def MNIST_train_test_14x14(root="../data"):
>>>>>>> 88b79a110fb5cd3d0ea75b8927ec339e7ff55a3d
    datasets.MNIST(root=root, train=True, download=True, transform=None)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((14, 14), interpolation=InterpolationMode.NEAREST)])
    dataset1 = datasets.MNIST(root=root, train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(root=root, train=False,
                              transform=transform)
    return dataset1, dataset2


def MNIST_train_test_14x14_norm(root="../data"):
    datasets.MNIST(root=root, train=True, download=True, transform=None)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize((14, 14), interpolation=InterpolationMode.NEAREST)])
    dataset1 = datasets.MNIST(root=root, train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(root=root, train=False,
                              transform=transform)
    return dataset1, dataset2


i = 5
data_train = MNIST_train_test_14x14()[0]
s=0
for i in range(1000):
    s+=torch.sum(data_train[i][0])

for i in range(20):
    print(data_train[i][1],torch.sum(data_train[i][0]),torch.sum(data_train[i][0]) / torch.sum(data_train[i][0] / 8))



import matplotlib.pyplot as plt



plt.subplot(1, 2, 1)
plt.imshow(data_train[1][0].squeeze(), vmax=1,cmap="gray")

# Plot 2:


plt.subplot(1, 2, 2)
plt.imshow((data_train[1][0].squeeze() / torch.sum(data_train[1][0] / 8)),vmax=1,cmap="gray")


plt.show()

def MNIST_train_test_9x9():
    datasets.MNIST(root='./data', train=True, download=True, transform=None)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((9, 9), interpolation=InterpolationMode.NEAREST)])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    return dataset1, dataset2


def encoding_to_spikes(data, time):
    return torch.bernoulli(data.repeat(time, 1, 1))

# print(print(torch.bernoulli(a[0][0])))
# print(a[1][0][::4, ::4])
# plt.imshow(torch.squeeze(a[1][0][::1, ::4, ::4]))
#
# print(encoding_to_spikes(a[0][0], 10))
# print(encoding_to_spikes(a[0][0], 10).size())

# print(type(b))
# print(torch.squeeze(a[1][0]))
# bb=torch.squeeze(a[1][0])
# print(bb[::2,::2].shape)
# plt.imshow(bb[::2,::2])

# for i in range(20):
#     print(a[i])
