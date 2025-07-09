import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random

def get_transforms():
    """Defines data augmentation and normalization for train and test sets."""
    # Training transforms include augmentation for better model generalization.
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 stats
    ])
    
    # Test transforms have no augmentation to ensure deterministic evaluation.
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 stats
    ])
    return train_transform, test_transform

def load_cifar100(train_transform, test_transform, val_ratio=0.1):
    """Loads CIFAR-100 and creates a train, validation, and test split."""
    trainset_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    # Split a portion of the training set for validation
    num_train = len(trainset_full)
    num_val = int(num_train * val_ratio)
    trainset, valset = random_split(trainset_full, [num_train - num_val, num_val])

    return trainset, valset, testset

def iid_split(dataset, num_clients):
    """Splits a dataset into IID (random and uniform) partitions for clients."""
    num_items_per_client = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    
    # Create a data subset for each client from the shuffled indices
    return [Subset(dataset, indices[i * num_items_per_client:(i + 1) * num_items_per_client]) for i in range(num_clients)]

def noniid_split(dataset, num_clients, num_classes_per_client):
    """Splits a dataset into non-IID partitions, where each client has data from a limited number of classes."""
    # Group all data indices by their class label
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(100)}
    
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, split its indices among the clients
    for class_idx in class_indices:
        data_for_class = class_indices[class_idx]
        np.random.shuffle(data_for_class)
        
        # Assign a limited number of classes to each client
        # This is a simplified approach; more complex distributions can be created
        assigned_clients = np.random.choice(num_clients, num_classes_per_client, replace=False)
        
        split_size = len(data_for_class) // num_classes_per_client
        for i, client_id in enumerate(assigned_clients):
            start = i * split_size
            end = start + split_size
            client_indices[client_id].extend(data_for_class[start:end])

    # Create subset objects for each client
    return [Subset(dataset, indices) for indices in client_indices]
