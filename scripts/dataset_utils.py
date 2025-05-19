import numpy as np
import random
from collections import defaultdict

def iid_split(dataset, num_clients):
    """
    Uniforme distribuzione delle classi tra i client.
    """
    data_per_client = len(dataset) // num_clients
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    return [all_indices[i * data_per_client: (i + 1) * data_per_client] for i in range(num_clients)]

def non_iid_split(dataset, num_clients, nc):
    """
    Ogni client riceve dati da Nc classi (non-iid).
    """
    targets = np.array(dataset.dataset.targets if hasattr(dataset, 'dataset') else dataset.targets)
    class_indices = defaultdict(list)

    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    for c in class_indices:
        np.random.shuffle(class_indices[c])

    class_pool = list(class_indices.keys())
    client_indices = [[] for _ in range(num_clients)]

    for client_id in range(num_clients):
        chosen_classes = random.sample(class_pool, nc)
        for cls in chosen_classes:
            n = len(class_indices[cls]) // num_clients
            client_indices[client_id] += class_indices[cls][:n]
            class_indices[cls] = class_indices[cls][n:]

    return client_indices
