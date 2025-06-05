import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import argparse
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

from data.cifar100_loader import get_transforms, load_cifar100, iid_split, noniid_split
from models.vit_dino import get_dino_vit_s16
from utils.logger import MetricLogger
from utils.checkpoint import save_checkpoint

def train_local(model, dataloader, criterion, optimizer, device, local_epochs):
    """
    Local training for a single client.
    """
    model.train()
    for _ in range(local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def aggregate_models(global_model, local_models):
    """
    Averages model parameters from all participating clients.
    """
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.stack([client_state[key].float() for client_state in local_models], dim=0).mean(dim=0)
    global_model.load_state_dict(global_state)
    return global_model

def evaluate(model, dataloader, criterion, device):
    """
    Global evaluation on the central test set.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), correct / total

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, test_tf = get_transforms()
    trainset, _, testset = load_cifar100(train_tf, test_tf)
    test_loader = DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False)

    # Shard the dataset among K clients
    if cfg["sharding"] == "iid":
        client_datasets = iid_split(trainset, cfg["K"])
    else:
        client_datasets = noniid_split(trainset, cfg["K"], cfg["Nc"])

    global_model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    logger = MetricLogger(cfg["log_path"])

    for round_num in range(1, cfg["rounds"] + 1):
        print(f"\n--- Round {round_num} ---")
        local_models = []
        selected_clients = torch.randperm(cfg["K"])[:int(cfg["K"] * cfg["C"])]

        for client_id in tqdm(selected_clients, desc="Clients training"):
            client_model = deepcopy(global_model)
            client_model.to(device)
            client_loader = DataLoader(client_datasets[client_id], batch_size=cfg["batch_size"], shuffle=True)
            optimizer = optim.SGD(client_model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
            local_state = train_local(client_model, client_loader, criterion, optimizer, device, cfg["J"])
            local_models.append(local_state)

        # Aggregate local models
        global_model = aggregate_models(global_model, local_models)

        # Evaluate on test data
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        logger.log({
            "round": round_num,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        if round_num % cfg["save_every"] == 0:
            save_checkpoint(global_model, None, None, round_num, cfg["checkpoint_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
