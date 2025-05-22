import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset
import random
from collections import defaultdict

# Set random seed for reproducibility
torch.manual_seed(42)

def get_transforms():
    """Return standard transforms for CIFAR-100."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    return transform

def load_cifar100(root="./data"):
    """Download and load CIFAR-100 training and test datasets."""
    transform = get_transforms()
    trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    return trainset, testset

def split_validation(trainset, val_ratio=0.1):
    """Split the trainset into train and validation subsets."""
    total_size = len(trainset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    return random_split(trainset, [train_size, val_size])

def iid_split(dataset, num_clients):
    """Split dataset i.i.d. among clients."""
    data_size = len(dataset)
    indices = list(range(data_size))
    random.shuffle(indices)
    split = [indices[i::num_clients] for i in range(num_clients)]
    return split

def non_iid_split(dataset, num_clients, num_classes_per_client):
    """Split dataset non-i.i.d. among clients."""
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    all_labels = list(label_to_indices.keys())
    random.shuffle(all_labels)

    labels_per_client = [
        all_labels[i::num_clients] for i in range(num_clients)
    ]

    for client_id, labels in enumerate(labels_per_client):
        labels = labels[:num_classes_per_client]
        for label in labels:
            chosen = label_to_indices[label][:len(label_to_indices[label]) // num_clients]
            client_indices[client_id].extend(chosen)
            label_to_indices[label] = label_to_indices[label][len(chosen):]
    
    return client_indices
