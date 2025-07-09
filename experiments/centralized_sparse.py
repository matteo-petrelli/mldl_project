import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import json
import shutil
from tqdm import tqdm

# Import model, data loader, and utility functions
from models.vit_dino import get_dino_vit_s16
from data.cifar100_loader import get_transforms, load_cifar100
from utils.logger import MetricLogger
from utils.checkpoint import load_checkpoint, save_checkpoint
from optimizer.sparseSGDM import SparseSGDM
from optimizer.mask_utils import (
    compute_fisher_diagonal,
    build_mask_by_sensitivity,
    build_mask_by_magnitude,
    build_mask_randomly
)

def resume_if_possible(cfg, model, optimizer, scheduler):
    """Resumes training from a checkpoint and log file if they exist."""
    log_path = cfg['log_path']
    logger = MetricLogger(log_path)
    start_epoch = 0

    # Resume logs if the log file exists
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                prev_metrics = json.load(f)
            logger.metrics = prev_metrics
            start_epoch = len(prev_metrics)
            print(f"[Logger] Resumed log from epoch {start_epoch}")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous logs: {e}")

    # Resume from checkpoint if a path is provided and it exists
    resume_path = cfg.get("checkpoint_drive_path") if os.path.exists(cfg.get("checkpoint_drive_path", "")) else cfg.get("checkpoint_path")
    if resume_path and os.path.exists(resume_path):
        checkpoint_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = max(start_epoch, checkpoint_epoch)
        print(f"[Checkpoint] Resumed from {resume_path} (epoch {checkpoint_epoch})")

    return start_epoch, logger

def mask_to_param_list(mask_dict, model):
    """
    Converts a dictionary of masks into a list aligned with model parameters.
    Unmasked parameters get an all-ones mask for normal updates.
    """
    param_masks = []
    for name, param in model.named_parameters():
        # Use the specific mask if available
        if name in mask_dict:
            param_masks.append(mask_dict[name].to(param.device))
        # Otherwise, use an all-ones mask (no sparsity)
        else:
            param_masks.append(torch.ones_like(param))
    return param_masks

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs a single training epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation/Test"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(dataloader), correct / total

def main(args):
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Set up device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and create DataLoaders
    train_tf, test_tf = get_transforms()
    trainset, valset, testset = load_cifar100(train_tf, test_tf)
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # Load model and define loss function
    model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the optimizer once with a temporary all-ones mask
    optimizer = SparseSGDM(
        model.parameters(),
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=cfg.get('weight_decay', 0.0),
        mask=mask_to_param_list({}, model) # Empty initial mask (all ones)
    )

    calibration_rounds = cfg.get("calibration_rounds", 1)
    print(f"### STARTING CALIBRATION PHASE: {calibration_rounds} ROUNDS ###")

    # === MASK CALIBRATION LOGIC ===
    for calib_round in range(calibration_rounds):
        print(f"\n--- Calibration Round {calib_round + 1}/{calibration_rounds} ---")
        
        # 1. Calculate the mask based on the current model state
        mask_rule = cfg.get("mask_calibration_rule", "sensitivity_most")
        print(f"Recalculating mask with rule: {mask_rule}")
        mask_dict = {}

        if "sensitivity" in mask_rule:
            fisher_calc_loader = DataLoader(trainset, batch_size=cfg.get("fisher_batch_size", 1), shuffle=True)
            fisher = compute_fisher_diagonal(model, fisher_calc_loader, criterion, device)
            pick_least = (mask_rule == "sensitivity_least")
            mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=pick_least)
        elif "magnitude" in mask_rule:
            pick_highest = (mask_rule == "magnitude_highest")
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=pick_highest)
        elif mask_rule == "random":
            mask_dict = build_mask_randomly(model, cfg["sparsity_ratio"])
        else:
            print("No valid mask rule. Proceeding with dense training.")

        # 2. Update the mask in the optimizer
        mask_list = mask_to_param_list(mask_dict, model)
        optimizer.mask = mask_list
        print("Mask updated in the optimizer.")

        # 3. Fine-tune for 1 epoch to influence the next calibration round
        # (skip this step in the last round)
        if calib_round < calibration_rounds - 1:
            print("Fine-tuning the model for 1 epoch with the current mask...")
            train_one_epoch(model, train_loader, criterion, optimizer, device)

    print("\n### CALIBRATION PHASE COMPLETE ###")
    print("Starting main fine-tuning with the final mask.")

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # Resume training if possible
    start_epoch, logger = resume_if_possible(cfg, model, optimizer, scheduler)
    
    best_val_acc = 0.0
    patience = cfg.get("early_stopping_patience", 5)  # default value
    patience_counter = 0

    # Main training loop
    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.log({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping activated at {epoch+1} (val_acc hasn't improved for {patience} epochs)")
            break

        # Standard checkpointing
        os.makedirs(os.path.dirname(cfg['checkpoint_path']), exist_ok=True)
        save_checkpoint(model, optimizer, scheduler, epoch + 1, path=cfg['checkpoint_path'])
        print(f"Local checkpoint saved: {cfg['checkpoint_path']}")
        
        # Backup checkpoint and logs to a secondary location (e.g., Google Drive)
        if cfg.get('checkpoint_drive_path'):
            os.makedirs(os.path.dirname(cfg['checkpoint_drive_path']), exist_ok=True)
            shutil.copy(cfg['checkpoint_path'], cfg['checkpoint_drive_path'])
            print(f"Drive backup checkpoint: {cfg['checkpoint_drive_path']}")
        
        if cfg.get("log_drive_path") and os.path.exists(cfg['log_path']):
            os.makedirs(os.path.dirname(cfg["log_drive_path"]), exist_ok=True)
            shutil.copy(cfg['log_path'], cfg["log_drive_path"])

    # Final evaluation on the test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
