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

# --- Custom Module Imports ---
from data.cifar100_loader import get_transforms, load_cifar100, iid_split, noniid_split
from models.vit_dino import get_dino_vit_s16
from utils.logger import MetricLogger
from utils.checkpoint import save_checkpoint, load_checkpoint

def resume_if_possible(cfg, model):
    """Resumes a federated learning run from a saved checkpoint and log file."""
    # Ensure log directory exists and initialize logger
    os.makedirs(os.path.dirname(cfg['log_path']), exist_ok=True)
    if not os.path.exists(cfg['log_path']):
        with open(cfg['log_path'], 'w') as f:
            json.dump([], f)
    logger = MetricLogger(save_path=cfg['log_path'])
    start_round = 1

    # Attempt to resume logs from the local file
    if os.path.exists(cfg['log_path']) and os.path.getsize(cfg['log_path']) > 0:
        try:
            with open(cfg['log_path'], 'r') as f:
                logger.metrics = json.load(f)
            start_round = len(logger.metrics) + 1
            print(f"[Logger] Resumed from round {start_round}")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous logs: {e}")

    # Find and load the latest checkpoint, prioritizing the drive path
    resume_path = None
    if cfg.get("checkpoint_drive_path") and os.path.exists(cfg["checkpoint_drive_path"]):
        resume_path = cfg["checkpoint_drive_path"]
    elif cfg.get("checkpoint_path") and os.path.exists(cfg["checkpoint_path"]):
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
    """Performs the local training loop for a single client."""
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
    """Averages model parameters from clients to update the global model (FedAvg)."""
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.stack([client_state[key].float() for client_state in local_models], dim=0).mean(dim=0)
    global_model.load_state_dict(global_state)
    return global_model

def evaluate(model, dataloader, criterion, device):
    """Evaluates the global model on the central test set."""
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
    # --- Configuration and Setup ---
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading and Sharding ---
    train_tf, test_tf = get_transforms()
    trainset, _, testset = load_cifar100(train_tf, test_tf)
    test_loader = DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False)

    # Split the training data among K clients
    if cfg["sharding"] == "iid":
        client_datasets = iid_split(trainset, cfg["K"])
    else:
        client_datasets = noniid_split(trainset, cfg["K"], cfg.get("Nc", 10))

    # --- Model and Logger Initialization ---
    global_model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    start_round, logger = resume_if_possible(cfg, global_model)

    # --- Federated Learning Rounds ---
    for round_num in range(start_round, cfg["rounds"] + 1):
        print(f"\n--- Round {round_num} ---")
        local_models = []
        # Select a fraction of clients to participate in this round
        selected_clients = torch.randperm(cfg["K"])[:int(cfg["K"] * cfg["C"])]

        for client_id in tqdm(selected_clients, desc="Clients training"):
            # Prepare the client model and data
            client_model = deepcopy(global_model).to(device)
            client_loader = DataLoader(client_datasets[client_id.item()], batch_size=cfg["batch_size"], shuffle=True)
            
            # Set up for linear probing: freeze backbone, train only the head
            for param in client_model.parameters():
                param.requires_grad = False
            for param in client_model.head.parameters():
                param.requires_grad = True
            
            # The optimizer will only update the unfrozen parameters (the head)
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, client_model.parameters()), 
                lr=cfg["lr"],
                momentum=0.9,
                weight_decay=cfg["weight_decay"]
            )
            
            # Use a scheduler for local training epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["J"])
        
            # Perform local training
            local_state = train_local(client_model, client_loader, criterion, optimizer, scheduler, device, cfg["J"])
            local_models.append(local_state)
            
        # --- Aggregation and Evaluation ---
        # Aggregate the trained local models into a new global model
        global_model = aggregate_models(global_model, local_models)

        # Evaluate the new global model's performance on the test set
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        logger.log({"round": round_num, "test_loss": test_loss, "test_acc": test_acc})

        # --- Checkpointing ---
        # Periodically save the global model and logs
        if round_num % cfg.get("save_every", 1) == 0:
            save_checkpoint(global_model, None, None, round_num, cfg["checkpoint_path"])
            print(f"[Checkpoint] Saved locally: {cfg['checkpoint_path']}")
        
            if "checkpoint_drive_path" in cfg:
                shutil.copy(cfg["checkpoint_path"], cfg["checkpoint_drive_path"])
                print(f"[Checkpoint] Backed up to Drive: {cfg['checkpoint_drive_path']}")
        
            if "log_drive_path" in cfg:
                if os.path.exists(cfg["log_path"]):
                    shutil.copy(cfg["log_path"], cfg["log_drive_path"])
                    print(f"[Log] Copied to Drive: {cfg['log_drive_path']}")
                else:
                    print(f"[Log Warning] Log file '{cfg['log_path']}' does not exist and was not copied.")

if __name__ == "__main__":
    # Script entry point: parse config and start the main training function
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args)
