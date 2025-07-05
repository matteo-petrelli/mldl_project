import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import os

from data.cifar100_loader import get_transforms, load_cifar100, noniid_split
from models.vit_dino import get_dino_vit_s16
from sparse.optimizer import SparseSGDM
from sparse.mask_utils import compute_fisher_diagonal, build_mask_by_sensitivity, build_mask_by_magnitude, build_mask_randomly
from utils.logger import MetricLogger

def mask_to_param_list(mask_dict, model):
    """
    Convert mask dict into list format matching model parameters (order matters).
    """
    param_masks = []
    i = 0
    for name, param in model.named_parameters():
        if name in mask_dict:
            param_masks.append(mask_dict[name].to(param.device))
        else:
            param_masks.append(torch.ones_like(param))
    return param_masks

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, test_tf = get_transforms()
    trainset, _, testset = load_cifar100(train_tf, test_tf)
    test_loader = DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False)

    # Split dataset for clients (even if simulating one client for sparse fine-tuning)
    client_datasets = noniid_split(trainset, cfg["K"], cfg["Nc"])
    # Define the DataLoader for the specific client being fine-tuned
    fine_tuning_train_loader = DataLoader(client_datasets[cfg["client_id"]], batch_size=cfg["batch_size"], shuffle=True)

    model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    
    mask_rule = cfg.get("mask_calibration_rule", "sensitivity_most") # Read the rule from config
    mask_dict = {}

    # --- Step 1 & 2: Calculate Fisher (if needed) and Create Mask ---
    # The Fisher calculation should only happen if a sensitivity-based rule is chosen.
    if "sensitivity" in mask_rule:
        print(f"Calculating Fisher Information for rule: {mask_rule}")
        fisher_calc_loader = DataLoader(client_datasets[0], batch_size=cfg["fisher_batch_size"], shuffle=False)
        fisher = compute_fisher_diagonal(model, fisher_calc_loader, criterion, device)

        if mask_rule == "sensitivity_most":
            # This is the original logic: update the most sensitive parameters
            mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=False)
        elif mask_rule == "sensitivity_least":
            # Update the least sensitive parameters
            mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=True)
        else:
            raise ValueError(f"Unrecognized sensitivity rule: {mask_rule}")


    elif "magnitude" in mask_rule:
        # For magnitude-based rules
        print(f"Creating mask based on magnitude: {mask_rule}")
        if mask_rule == "magnitude_highest":
            # Update parameters with highest magnitude
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=True)
        elif mask_rule == "magnitude_lowest":
            # Update parameters with lowest magnitude
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=False)
        else:
            raise ValueError(f"Unrecognized magnitude rule: {mask_rule}")

    elif mask_rule == "random":
        # For random rule
        print("Creating random mask.")
        mask_dict = build_mask_randomly(model, cfg["sparsity_ratio"])
    else:
        raise ValueError(f"Unrecognized mask calibration rule: {mask_rule}")

    # Convert the mask dictionary to an ordered list for the optimizer
    mask_list = mask_to_param_list(mask_dict, model)
    
    # Step 3: Fine-tuning with Mask
    # Use the DataLoader specifically prepared for fine-tuning
    optimizer = SparseSGDM(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=0.0005, mask=mask_list)
    logger = MetricLogger(cfg["log_path"])

    # Optional: Log to Google Drive if paths are provided in config
    if cfg.get("log_drive_path"):
        # You might want to create a separate logger for drive or integrate this into MetricLogger
        print(f"Also logging metrics to Google Drive at: {cfg['log_drive_path']}")

    model.train()
    for epoch in range(cfg["epochs"]):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"[Epoch {epoch+1}] Acc: {acc:.4f}")
        logger.log({"epoch": epoch+1, "test_acc": acc})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
