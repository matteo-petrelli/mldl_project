import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random

def get_transforms():
    """Defines and returns data transformations for training and testing."""
    # Defines augmentations (crop, flip) and normalization for the training set.
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    # Defines normalization for the test set (no augmentation).
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    return train_transform, test_transform

def load_cifar100(train_transform, test_transform, val_ratio=0.1):
    """Loads the CIFAR-100 dataset and creates train, validation, and test splits."""
    trainset_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    # Splits a fraction of the training data off for validation.
    num_train = len(trainset_full)
    num_val = int(num_train * val_ratio)
    num_train -= num_val
    trainset, valset = random_split(trainset_full, [num_train, num_val])

    return trainset, valset, testset

def iid_split(dataset, num_clients):
    """Splits a dataset into IID (randomly uniform) partitions for clients."""
    data_per_client = len(dataset) // num_clients
    # Randomly shuffles all data indices.
    indices = np.random.permutation(len(dataset))
    # Deals out equal-sized, random chunks to each client.
    return [Subset(dataset, indices[i * data_per_client:(i + 1) * data_per_client]) for i in range(num_clients)]

def noniid_split(dataset, num_clients, num_classes_per_client):
    """Splits a dataset into non-IID partitions based on class labels."""
    # Group all data indices by their class label.
    class_indices = {i: [] for i in range(100)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    for key in class_indices:
        random.shuffle(class_indices[key])

    client_data = [[] for _ in range(num_clients)]
    class_pool = list(range(100))
    # For each client...
    for client in range(num_clients):
        # ...assign a random subset of classes.
        assigned_classes = np.random.choice(class_pool, num_classes_per_client, replace=False)
        # For each assigned class...
        for cls in assigned_classes:
            if class_indices[cls]:
                # ...give the client a chunk of data from that class.
                chunk_size = len(class_indices[cls]) // num_clients
                client_data[client].extend(class_indices[cls][:chunk_size])
                # Remove the assigned chunk from the pool.
                class_indices[cls] = class_indices[cls][chunk_size:]

    return [Subset(dataset, indices) for indices in client_data]
