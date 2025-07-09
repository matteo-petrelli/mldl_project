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
from optimizer.sparseSGDM import SparseSGDM
from optimizer.mask_utils import (
    compute_fisher_diagonal,
    build_mask_by_sensitivity,
    build_mask_by_magnitude,
    build_mask_randomly
)

def aggregate_models(global_model, local_models):
    """Averages the state dictionaries of local models to update the global model (FedAvg)."""
    global_state = global_model.state_dict()
    for key in global_state.keys():
        # Average the parameters from all local models for each layer
        global_state[key] = torch.stack([client_state[key].float() for client_state in local_models], dim=0).mean(dim=0)
    global_model.load_state_dict(global_state)
    return global_model

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset, returning loss and accuracy."""
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

def mask_to_param_list(mask_dict, model):
    """Converts a dictionary of masks to a list aligned with model parameters."""
    param_masks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Use the specific mask if available, otherwise use a default all-ones mask
            mask = mask_dict.get(name, torch.ones_like(param))
            param_masks.append(mask.to(param.device))
    return param_masks

def resume_if_possible(cfg, model):
    """Resumes a federated learning run from a saved checkpoint and log file."""
    logger = MetricLogger(save_path=cfg['log_path'])
    start_round = 1
    
    # Attempt to resume logs
    if os.path.exists(cfg['log_path']) and os.path.getsize(cfg['log_path']) > 0:
        try:
            with open(cfg['log_path'], 'r') as f:
                logger.metrics = json.load(f)
            start_round = len(logger.metrics) + 1
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
            # Load only the model state; optimizer is created locally on clients
            checkpoint_round = load_checkpoint(resume_path, model, optimizer=None, scheduler=None)
            start_round = max(start_round, checkpoint_round + 1)
            print(f"[Checkpoint] Resumed from round {start_round}")
        except Exception as e:
            print(f"[Checkpoint Warning] Failed to load checkpoint: {e}")
            
    return start_round, logger

def train_local_sparse(model, dataloader, criterion, device, cfg, existing_mask=None):
    """Performs local training on a client using a sparse optimizer."""
    mask_dict = {}
    if existing_mask is not None:
        # Use the fixed mask provided from a previous round
        mask_dict = existing_mask
    else:
        # Compute a new mask based on the specified rule (e.g., sensitivity, magnitude)
        mask_rule = cfg.get("mask_calibration_rule", "random")
        if "sensitivity" in mask_rule:
            fisher_dataloader = DataLoader(dataloader.dataset, batch_size=1, shuffle=True)
            fisher = compute_fisher_diagonal(model, fisher_dataloader, criterion, device)
            mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=("least" in mask_rule))
        elif "magnitude" in mask_rule:
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=("highest" in mask_rule))
        elif mask_rule == "random":
            mask_dict = build_mask_randomly(model, cfg["sparsity_ratio"])
    
    # Prepare the mask and initialize the sparse optimizer
    mask_list = mask_to_param_list(mask_dict, model)
    optimizer = SparseSGDM(model.parameters(), lr=cfg["lr"], momentum=0.9, mask=mask_list)
    
    # Perform local training for J epochs
    model.train()
    for _ in range(cfg["J"]):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model.state_dict(), mask_dict

def main(args):
    # --- Configuration and Setup ---
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading and Sharding ---
    train_tf, test_tf = get_transforms()
    trainset, _, testset = load_cifar100(train_tf, test_tf, val_ratio=0.0)
    test_loader = DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    # Split the training data among clients based on the specified sharding strategy
    if cfg["sharding"] == "iid":
        client_datasets = iid_split(trainset, cfg["K"])
    else:
        client_datasets = noniid_split(trainset, cfg["K"], cfg.get("Nc", 10))

    # --- Model and Logger Initialization ---
    global_model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    start_round, logger = resume_if_possible(cfg, global_model)

    client_masks = {}
    # Number of initial rounds for mask calibration
    calibration_rounds = cfg.get("calibration_rounds", cfg["rounds"])

    # --- Federated Learning Rounds ---
    for round_num in range(start_round, cfg["rounds"] + 1):
        # Determine if the current round is for calibration or fine-tuning
        is_calibration_round = (round_num <= calibration_rounds)

        print(f"\n--- Round {round_num}/{cfg['rounds']} ---")
        if is_calibration_round:
            print("Mode: INITIAL CALIBRATION PHASE")
        else:
            print("Mode: FINE-TUNING (using fixed masks)")

        # Select a fraction of clients for this round and train them
        local_models = []
        selected_clients = torch.randperm(cfg["K"])[:int(cfg["K"] * cfg["C"])]
        for client_id_tensor in tqdm(selected_clients, desc="Training clients"):
            client_id = client_id_tensor.item()
            client_model = deepcopy(global_model).to(device)
            client_loader = DataLoader(client_datasets[client_id], batch_size=cfg["batch_size"], shuffle=True)
            
            # Use a pre-computed mask if we are past the calibration phase
            mask_to_use = None if is_calibration_round else client_masks.get(client_id)
            
            # Perform local training
            local_state, used_mask = train_local_sparse(client_model, client_loader, criterion, device, cfg, existing_mask=mask_to_use)
            local_models.append(local_state)
            
            # Store the computed mask for this client to be reused later
            client_masks[client_id] = used_mask
            
        # --- Aggregation and Evaluation ---
        global_model = aggregate_models(global_model, local_models)
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Global Model Test Accuracy: {test_acc*100:.2f}%")
        logger.log({ "round": round_num, "test_loss": test_loss, "test_acc": test_acc })

        # --- Checkpointing ---
        if round_num % cfg.get("save_every", 10) == 0:
            save_checkpoint(global_model, None, None, round_num, cfg["checkpoint_path"])
            print(f"[Checkpoint] Saved locally: {cfg['checkpoint_path']}")
            if "checkpoint_drive_path" in cfg:
                shutil.copy(cfg["checkpoint_path"], cfg["checkpoint_drive_path"])
                print(f"[Checkpoint] Backed up to Drive: {cfg['checkpoint_drive_path']}")
            if "log_drive_path" in cfg:
                shutil.copy(cfg["log_path"], cfg["log_drive_path"])
                print(f"[Log] Copied to Drive: {cfg['log_drive_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args)
