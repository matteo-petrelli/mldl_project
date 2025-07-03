import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import argparse
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
import json
import shutil

from data.cifar100_loader import get_transforms, load_cifar100, iid_split, noniid_split
from models.vit_dino import get_dino_vit_s16
from utils.logger import MetricLogger
from utils.checkpoint import save_checkpoint, load_checkpoint

import os
import json

def resume_if_possible(cfg, model):
    """
    Resume training from checkpoint and logs. Prefer Drive paths if available.
    """
    local_log_path = cfg['log_path']

    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # If the file doesn't exist, initialize it as empty
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            json.dump([], f)

    # Initialize logger
    logger = MetricLogger(save_path=log_path)
    start_round = 1

    # Resume logger if the file is valid
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        try:
            with open(log_path, 'r') as f:
                prev_metrics = json.load(f)
            logger.metrics = prev_metrics
            start_round = len(prev_metrics) + 1
            print(f"[Logger] Resumed from round {start_round}")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous logs: {e}")

    # Resume model checkpoint (Drive has priority)
    resume_path = None
    if os.path.exists(cfg.get("checkpoint_drive_path", "")):
        resume_path = cfg["checkpoint_drive_path"]
    elif os.path.exists(cfg.get("checkpoint_path", "")):
        resume_path = cfg["checkpoint_path"]

    if resume_path:
        try:
            checkpoint_round = load_checkpoint(resume_path, model, optimizer=None, scheduler=None)
            start_round = max(start_round, checkpoint_round + 1)
            print(f"[Checkpoint] Resumed from round {checkpoint_round} ({resume_path})")
        except Exception as e:
            print(f"[Checkpoint Warning] Failed to load checkpoint: {e}")

    return start_round, logger

def train_local(model, dataloader, criterion, optimizer, scheduler, device, local_epochs):
    """
    Local training for a single client with cosine annealing LR scheduler.
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
        if scheduler:
            scheduler.step()
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

    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
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

    start_round, logger = resume_if_possible(cfg, global_model)

    for round_num in range(start_round, cfg["rounds"] + 1):
        print(f"\n--- Round {round_num} ---")
        local_models = []
        selected_clients = torch.randperm(cfg["K"])[:int(cfg["K"] * cfg["C"])]

        for client_id in tqdm(selected_clients, desc="Clients training"):
            client_model = deepcopy(global_model)
            client_model.to(device)
            
            for param in client_model.parameters():
                param.requires_grad = False
        
            for param in client_model.head.parameters():
                param.requires_grad = True
            
            client_loader = DataLoader(client_datasets[client_id], batch_size=cfg["batch_size"], shuffle=True)
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, client_model.parameters()), 
                lr=cfg["lr"],
                momentum=0.9,
                weight_decay=cfg["weight_decay"]
            )
            
            optimizer = optim.SGD(client_model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
        
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["J"])
        
            local_state = train_local(client_model, client_loader, criterion, optimizer, scheduler, device, cfg["J"])
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
            os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
            save_checkpoint(global_model, None, None, round_num, cfg["checkpoint_path"])
            print(f"[Checkpoint] Salvato localmente: {cfg['checkpoint_path']}")
        
            if "checkpoint_drive_path" in cfg:
                os.makedirs(os.path.dirname(cfg["checkpoint_drive_path"]), exist_ok=True)
                shutil.copy(cfg["checkpoint_path"], cfg["checkpoint_drive_path"])
                print(f"[Checkpoint] Backup su Drive: {cfg['checkpoint_drive_path']}")
        
            if "log_drive_path" in cfg:
                os.makedirs(os.path.dirname(cfg["log_drive_path"]), exist_ok=True)
        
                if os.path.exists(cfg["log_path"]):
                    shutil.copy(cfg["log_path"], cfg["log_drive_path"])
                    print(f"[Log] Copiato su Drive: {cfg['log_drive_path']}")
                else:
                    print(f"[Log Warning] Il file di log '{cfg['log_path']}' non esiste e non è stato copiato.")

        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
