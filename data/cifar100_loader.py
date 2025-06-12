import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    return train_transform, test_transform

def load_cifar100(train_transform, test_transform, val_ratio=0.1):
    trainset_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    num_train = len(trainset_full)
    num_val = int(num_train * val_ratio)
    num_train -= num_val
    trainset, valset = random_split(trainset_full, [num_train, num_val])

    return trainset, valset, testset

def iid_split(dataset, num_clients):
    data_per_client = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    return [Subset(dataset, indices[i * data_per_client:(i + 1) * data_per_client]) for i in range(num_clients)]

def noniid_split(dataset, num_clients, num_classes_per_client):
    class_indices = {i: [] for i in range(100)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    for key in class_indices:
        random.shuffle(class_indices[key])

    client_data = [[] for _ in range(num_clients)]
    class_pool = list(range(100))
    for client in range(num_clients):
        assigned_classes = np.random.choice(class_pool, num_classes_per_client, replace=False)
        for cls in assigned_classes:
            if class_indices[cls]:
                client_data[client].extend(class_indices[cls][:len(class_indices[cls]) // num_clients])
                class_indices[cls] = class_indices[cls][len(class_indices[cls]) // num_clients:]

    return [Subset(dataset, indices) for indices in client_data]
