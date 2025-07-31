import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

from src.data.utils import get_target_label_idx, global_contrast_normalization, OneClass


def MNIST_loader(train_batch, test_batch, Class):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)

    Digits, new_test, labels = OneClass(train_dataset, test_dataset, Class)
    train_loader = torch.utils.data.DataLoader(
        Digits, batch_size=train_batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        new_test, batch_size=test_batch, shuffle=False, num_workers=2)
    return train_loader, test_loader, labels


def FMNIST_loader(train_batch, test_batch, Class):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)

    Digits, new_test, labels = OneClass(train_dataset, test_dataset, Class)
    train_loader = torch.utils.data.DataLoader(
        Digits, batch_size=train_batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        new_test, batch_size=test_batch, shuffle=False, num_workers=2)
    return train_loader, test_loader, labels


def CIFAR_loader(train_batch, test_batch, Class):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l2'))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    Digits, new_test, labels = OneClass(train_dataset, test_dataset, Class)
    train_loader = torch.utils.data.DataLoader(
        Digits, batch_size=train_batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        new_test, batch_size=test_batch, shuffle=False, num_workers=2)
    return train_loader, test_loader, labels


def Speech_loader(train_batch, test_batch):
    X = pd.read_csv('data/speech.csv', header=None)
    y = X[400].copy()
    X.drop(columns=400, inplace=True)

    R = int((3625 + 61) * 0.1)
    X_test = X[:R]
    X_train = X[R:]
    y_test = y[:R]
    y_train = y[R:]

    train = torch.utils.data.TensorDataset(torch.Tensor(X_train.values))
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True, drop_last=True)
    test = torch.utils.data.TensorDataset(torch.Tensor(X_test.values), torch.Tensor(y_test.values))
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=False)
    return train_loader, test_loader, y_test


def PIMA_loader(train_batch, test_batch):
    X = pd.read_csv('data/pima.csv', header=None)
    X.sort_values(by=8, inplace=True, ascending=False)
    y = X[8].copy()
    X.drop(columns=8, inplace=True)

    R = int(768 * 0.4)
    X_test = X[:R]
    X_train = X[R:]
    y_test = y[:R]
    y_train = y[R:]

    train = torch.utils.data.TensorDataset(torch.Tensor(X_train.values))
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch, shuffle=True, drop_last=True)
    test = torch.utils.data.TensorDataset(torch.Tensor(X_test.values), torch.Tensor(y_test.values))
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch, shuffle=False)
    return train_loader, test_loader, y_test
