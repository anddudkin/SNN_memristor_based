import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode


# print(MNIST)

def rand_in_U(n_in_neurons):  # [1., 1., 0., 0., 1., 1., 1., 0., 1., 0.]
    U = torch.bernoulli(torch.randn(n_in_neurons).uniform_())
    return U


def rand_values():
    G = torch.rand((40, 40))
    U = torch.rand(40)
    return [U, G]


def MNIST_train_test():
    datasets.MNIST(root='./data', train=False, download=True, transform=None)

    transform = transforms.Compose([
        transforms.ToTensor()])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    # print(dataset1[0])
    # print(dataset2)

    # plt.imshow(dataset1.data[0])
    # plt.title(dataset1[0][1])
    # plt.show()
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1)
    test_loader = torch.utils.data.DataLoader(dataset2)
    # print(train_loader)
    return [train_loader, test_loader, dataset1, dataset2]


# for batch_idx, (data, target) in enumerate(train_loader):
#     print(data, target)

def MNIST_train_test_14x14():
    datasets.MNIST(root='./data', train=False, download=True, transform=None)

    transform = transforms.Compose([
        transforms.ToTensor(),  transforms.Resize((14, 14))])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    return dataset1, dataset2


def MNIST_train_test_9x9():
    datasets.MNIST(root='./data', train=False, download=True, transform=None)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((9, 9), antialias=False)])
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
