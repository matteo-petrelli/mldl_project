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
import matplotlib.pyplot as plt 

# Import project-specific modules
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

def generate_and_save_plot(metrics, config_filename):
    """
    Generates and saves a plot with federated training results.
    The plot is saved in a 'plots' directory created in the current folder.
    """
    # Check if there is data to plot
    if not metrics:
        print("Warning: No metrics data to plot.")
        return

    # --- Automatic Path and Filename Generation ---
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    base_name = os.path.basename(config_filename)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(plots_dir, f"{name_without_ext}.png")

    # --- Data Extraction for Federated Learning ---
    # The x-axis is 'round' for federated learning
    rounds = [m['round'] for m in metrics]
    test_acc = [m['test_acc'] for m in metrics]
    test_loss = [m['test_loss'] for m in metrics]

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Federated Training Results for: {name_without_ext}", fontsize=16, weight='bold')

    # Plot Test Accuracy
    ax1.plot(rounds, test_acc, 'o-', label='Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Accuracy Trend', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Test Loss
    ax2.plot(rounds, test_loss, 'o--', label='Test Loss')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss Trend', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot successfully saved to: {output_path}")
    plt.close()

def aggregate_models(global_model, local_models):
    """Averages the state dictionaries of local models to update the global model."""
    global_state = global_model.state_dict()
    for key in global_state.keys():
        # Average the parameters from all local models
        global_state[key] = torch.stack([client_state[key].float() for client_state in local_models], dim=0).mean(dim=0)
    global_model.load_state_dict(global_state)
    return global_model
    

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset."""
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
            # Use the specific mask if available, otherwise use a mask of all ones
            if name in mask_dict:
                param_masks.append(mask_dict[name].to(param.device))
            else:
                param_masks.append(torch.ones_like(param))
    return param_masks

def resume_if_possible(cfg, model):
    """Resumes training from a checkpoint and log file if they exist."""
    local_log_path = cfg['log_path']
    os.makedirs(os.path.dirname(local_log_path), exist_ok=True)
    # Ensure log file exists
    if not os.path.exists(local_log_path):
        with open(local_log_path, 'w') as f:
            json.dump([], f)
            
    logger = MetricLogger(save_path=local_log_path)
    start_round = 1
    
    # Attempt to load previous logs
    if os.path.exists(local_log_path) and os.path.getsize(local_log_path) > 0:
        try:
            with open(local_log_path, 'r') as f:
                prev_metrics = json.load(f)
            logger.metrics = prev_metrics
            start_round = len(prev_metrics) + 1
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous local logs: {e}")
            
    # Determine the correct checkpoint path to resume from
    resume_path = None
    if cfg.get("checkpoint_drive_path") and os.path.exists(cfg["checkpoint_drive_path"]):
        resume_path = cfg["checkpoint_drive_path"]
    elif cfg.get("checkpoint_path") and os.path.exists(cfg["checkpoint_path"]):
        resume_path = cfg["checkpoint_path"]
        
    # Load the checkpoint if found
    if resume_path:
        try:
            checkpoint_round = load_checkpoint(resume_path, model, optimizer=None, scheduler=None)
            start_round = max(start_round, checkpoint_round + 1)
        except Exception as e:
            print(f"[Checkpoint Warning] Failed to load checkpoint: {e}")
            
    return start_round, logger

def train_local_sparse(model, dataloader, criterion, device, cfg, existing_mask=None):
    """Performs local training on a client with sparse updates."""
    mask_dict = {}
    if existing_mask is not None:
        # Use the provided mask if in the fine-tuning phase
        mask_dict = existing_mask
    else:
        # Otherwise, compute a new mask based on the specified rule
        mask_rule = cfg.get("mask_calibration_rule")
        if "sensitivity" in mask_rule:
            fisher_dataloader = DataLoader(dataloader.dataset, batch_size=1, shuffle=True)
            fisher = compute_fisher_diagonal(model, fisher_dataloader, criterion, device)
            pick_least = (mask_rule == "sensitivity_least")
            mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=pick_least)
        elif "magnitude" in mask_rule:
            pick_highest = (mask_rule == "magnitude_highest")
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=pick_highest)
        elif mask_rule == "random":
            mask_dict = build_mask_randomly(model, cfg["sparsity_ratio"])
    
    # Create the sparse optimizer with the computed mask
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
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Setup environment
    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, test_tf = get_transforms()
    trainset, _, testset = load_cifar100(train_tf, test_tf, val_ratio=0.0)
    test_loader = DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    # Split data among clients
    if cfg["sharding"] == "iid":
        client_datasets = iid_split(trainset, cfg["K"])
    else:
        client_datasets = noniid_split(trainset, cfg["K"], cfg.get("Nc", 10))

    # Initialize global model, criterion, and resume state
    global_model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    start_round, logger = resume_if_possible(cfg, global_model)

    client_masks = {}
    # Determine the number of initial calibration rounds
    # If not specified, every round is a calibration round (original behavior)
    calibration_rounds = cfg.get("calibration_rounds", cfg["rounds"])

    for round_num in range(start_round, cfg["rounds"] + 1):
        # Check if the current round is for calibration or fine-tuning
        is_calibration_round = (round_num <= calibration_rounds)

        print(f"\n--- Round {round_num}/{cfg['rounds']} ---")
        if is_calibration_round:
            print("Mode: INITIAL CALIBRATION PHASE")
        else:
            print("Mode: FINE-TUNING (using fixed masks)")

        local_models = []
        # Randomly select a fraction C of clients
        selected_clients = torch.randperm(cfg["K"])[:int(cfg["K"] * cfg["C"])]

        for client_id_tensor in tqdm(selected_clients, desc="Clients training"):
            client_id = client_id_tensor.item()
            client_model = deepcopy(global_model)
            client_model.to(device)
            client_loader = DataLoader(client_datasets[client_id], batch_size=cfg["batch_size"], shuffle=True)
            
            # Decide whether to use a pre-computed mask
            mask_to_use = None
            if not is_calibration_round:
                mask_to_use = client_masks.get(client_id, None)
            
            # Perform local training
            local_state, used_mask = train_local_sparse(client_model, client_loader, criterion, device, cfg, existing_mask=mask_to_use)
            local_models.append(local_state)
            
            # Save the mask for the client.
            # After calibration, this simply re-saves the same mask.
            client_masks[client_id] = used_mask
            
        # Aggregate local models and evaluate the new global model
        global_model = aggregate_models(global_model, local_models)
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        logger.log({ "round": round_num, "test_loss": test_loss, "test_acc": test_acc })

        # Save checkpoint periodically
        if round_num % cfg.get("save_every", 10) == 0:
            os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
            save_checkpoint(global_model, None, None, round_num, cfg["checkpoint_path"])
            print(f"[Checkpoint] Saved locally: {cfg['checkpoint_path']}")
            
            # Backup to a secondary location if specified
            if "checkpoint_drive_path" in cfg:
                os.makedirs(os.path.dirname(cfg["checkpoint_drive_path"]), exist_ok=True)
                shutil.copy(cfg["checkpoint_path"], cfg["checkpoint_drive_path"])
                print(f"[Checkpoint] Backed up to Drive: {cfg['checkpoint_drive_path']}")
                
            if "log_drive_path" in cfg:
                os.makedirs(os.path.dirname(cfg["log_drive_path"]), exist_ok=True)
                if os.path.exists(cfg["log_path"]):
                    shutil.copy(cfg["log_path"], cfg["log_drive_path"])
                    print(f"[Log] Copied to Drive: {cfg['log_drive_path']}")
                else:
                    print(f"[Log Warning] Log file '{cfg['log_path']}' does not exist and was not copied.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
