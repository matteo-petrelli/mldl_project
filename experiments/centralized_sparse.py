import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import json
import shutil
from tqdm import tqdm

# --- Import custom modules ---
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
    """Resumes training from a saved checkpoint and log file if they exist."""
    log_path = cfg['log_path']
    logger = MetricLogger(log_path)
    start_epoch = 0

    # Attempt to resume logs from a previous session
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                logger.metrics = json.load(f)
            start_epoch = len(logger.metrics)
            print(f"[Logger] Resumed log from epoch {start_epoch}")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous logs: {e}")

    # Attempt to resume model, optimizer, and scheduler from a checkpoint
    resume_path = cfg.get("checkpoint_drive_path") if os.path.exists(cfg.get("checkpoint_drive_path", "")) else cfg.get("checkpoint_path")
    if resume_path and os.path.exists(resume_path):
        checkpoint_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = max(start_epoch, checkpoint_epoch)
        print(f"[Checkpoint] Resumed from {resume_path} (epoch {checkpoint_epoch})")

    return start_epoch, logger

def mask_to_param_list(mask_dict, model):
    """Converts a dictionary of masks to a list aligned with model parameters."""
    param_masks = []
    for name, param in model.named_parameters():
        # Use the specific mask if available, otherwise use a default all-ones mask (no sparsity)
        mask = mask_dict.get(name, torch.ones_like(param))
        param_masks.append(mask.to(param.device))
    return param_masks

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs a single training epoch and returns the average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward and backward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset, returning average loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation/Test"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return total_loss / len(dataloader), correct / total

def main(args):
    # Load configuration from the specified YAML file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup device, data, and dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, test_tf = get_transforms()
    trainset, valset, testset = load_cifar100(train_tf, test_tf)
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # Load the DINO ViT model and define loss and optimizer
    model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    # Initialize the sparse optimizer with a temporary, all-ones mask
    optimizer = SparseSGDM(
        model.parameters(),
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=cfg.get('weight_decay', 0.0),
        mask=mask_to_param_list({}, model) # Initial empty mask results in all ones
    )

    # --- Mask Calibration Phase ---
    # This phase iteratively calculates and refines the sparsity mask before main training.
    calibration_rounds = cfg.get("calibration_rounds", 1)
    print(f"### STARTING MASK CALIBRATION PHASE: {calibration_rounds} ROUNDS ###")

    for calib_round in range(calibration_rounds):
        print(f"\n--- Calibration Round {calib_round + 1}/{calibration_rounds} ---")
        
        # 1. Recalculate the mask based on the current model state and config rule
        mask_rule = cfg.get("mask_calibration_rule", "sensitivity_most")
        print(f"Recalculating mask with rule: {mask_rule}")
        mask_dict = {}

        if "sensitivity" in mask_rule:
            fisher_loader = DataLoader(trainset, batch_size=cfg.get("fisher_batch_size", 1), shuffle=True)
            fisher_info = compute_fisher_diagonal(model, fisher_loader, criterion, device)
            pick_least = (mask_rule == "sensitivity_least")
            mask_dict = build_mask_by_sensitivity(fisher_info, cfg["sparsity_ratio"], pick_least_sensitive=pick_least)
        elif "magnitude" in mask_rule:
            pick_highest = (mask_rule == "magnitude_highest")
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=pick_highest)
        elif mask_rule == "random":
            mask_dict = build_mask_randomly(model, cfg["sparsity_ratio"])
        else:
            print("No valid mask rule specified. Proceeding with dense training.")

        # 2. Update the optimizer with the newly computed mask
        optimizer.mask = mask_to_param_list(mask_dict, model)
        print("Mask updated in the optimizer.")

        # 3. Fine-tune for one epoch to influence the next calibration round
        if calib_round < calibration_rounds - 1:
            print("Fine-tuning model for 1 epoch with the current mask...")
            train_one_epoch(model, train_loader, criterion, optimizer, device)

    print("\n### MASK CALIBRATION PHASE COMPLETE ###")
    print("Starting main fine-tuning with the final mask.")

    # Initialize the learning rate scheduler and resume if possible
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    start_epoch, logger = resume_if_possible(cfg, model, optimizer, scheduler)
    
    # Setup for early stopping
    best_val_acc = 0.0
    patience = cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    # --- Main Training Loop ---
    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        
        # Train, validate, and update learning rate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.log({ "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc })
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping activated at epoch {epoch+1}")
            break

        # Save checkpoint locally and to a backup location
        save_checkpoint(model, optimizer, scheduler, epoch + 1, path=cfg['checkpoint_path'])
        shutil.copy(cfg['checkpoint_path'], cfg['checkpoint_drive_path'])
        print(f"Local checkpoint saved: {cfg['checkpoint_path']}")
        
    # Final evaluation on the test set after training is complete
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    # Parse command-line arguments to get the config file path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
