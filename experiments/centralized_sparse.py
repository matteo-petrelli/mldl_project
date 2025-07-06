import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import json
import shutil
from tqdm import tqdm

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
    log_path = cfg.get('log_drive_path', cfg['log_path'])
    logger = MetricLogger(log_path)
    start_epoch = 0

    # Resume logs
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                prev_metrics = json.load(f)
            logger.metrics = prev_metrics
            start_epoch = len(prev_metrics)
            print(f"[Logger] Resumed log from epoch {start_epoch}")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous logs: {e}")

    # Resume checkpoint
    resume_path = cfg.get("checkpoint_drive_path") if os.path.exists(cfg.get("checkpoint_drive_path", "")) else cfg.get("checkpoint_path")
    if resume_path and os.path.exists(resume_path):
        checkpoint_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = max(start_epoch, checkpoint_epoch)
        print(f"[Checkpoint] Resumed from {resume_path} (epoch {checkpoint_epoch})")

    return start_epoch, logger

# Function to convert the mask dictionary into a list suitable for the optimizer
def mask_to_param_list(mask_dict, model):
    """
    Converts a dictionary of masks (param_name: mask_tensor) into a list
    of mask tensors aligned with model's parameters.
    Parameters not found in mask_dict will have an all-ones mask, meaning they are updated normally.
    """
    param_masks = []
    for name, param in model.named_parameters():
        if name in mask_dict:
            # If a mask exists for this parameter, use it
            param_masks.append(mask_dict[name].to(param.device))
        else:
            # If a parameter is not in the mask_dict, it means it's not subject to sparsity.
            # Its mask should be all ones, effectively not masking any gradients.
            param_masks.append(torch.ones_like(param))
    return param_masks

# The `train_one_epoch` and `evaluate` functions remain identical to your provided ones
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs a single training epoch.
    """
    model.train() # Set the model to training mode
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device) # Move data to device
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        optimizer.zero_grad() # Clear gradients
        loss.backward() # Backward pass to compute gradients
        optimizer.step() # Update model parameters
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1) # Get the predicted classes
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # Count correct predictions
    return total_loss / len(dataloader), correct / total # Return average loss and accuracy

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataset.
    """
    model.eval() # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculations
        for images, labels in tqdm(dataloader, desc="Validation/Test"):
            images, labels = images.to(device), labels.to(device) # Move data to device
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1) # Get the predicted classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # Count correct predictions
    return total_loss / len(dataloader), correct / total # Return average loss and accuracy


def main(args):
    # Load configuration from the specified YAML file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Set up the device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data transformations and load the CIFAR-100 dataset
    train_tf, test_tf = get_transforms()
    trainset, valset, testset = load_cifar100(train_tf, test_tf)

    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # === 2. MODEL LOADING MODIFICATION ===
    # Load the DINO ViT-S/16 model with the specified number of output classes.
    # All parameters are initially trainable for fine-tuning.
    model = get_dino_vit_s16(num_classes=100).to(device)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # === 3. INSERT MASK CALIBRATION LOGIC ===
    # Retrieve the mask calibration rule from the configuration, defaulting to "sensitivity_most"
    mask_rule = cfg.get("mask_calibration_rule", "sensitivity_most")
    mask_dict = {}

    print(f"🛠️ Calibrating mask using rule: {mask_rule}")
    if "sensitivity" in mask_rule:
        # Create a DataLoader for Fisher Information calculation 
        fisher_calc_loader = DataLoader(trainset, batch_size=cfg.get("fisher_batch_size", 1), shuffle=False)
        # Compute the diagonal Fisher Information Matrix
        fisher = compute_fisher_diagonal(model, fisher_calc_loader, criterion, device)
        # Determine if we should pick the least sensitive parameters
        pick_least = (mask_rule == "sensitivity_least")
        # Build the mask based on sensitivity
        mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=pick_least)

    elif "magnitude" in mask_rule:
        # Determine if we should pick parameters with the highest magnitude
        pick_highest = (mask_rule == "magnitude_highest")
        # Build the mask based on parameter magnitudes
        mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=pick_highest)

    elif mask_rule == "random":
        # Build a random mask
        mask_dict = build_mask_randomly(model, cfg["sparsity_ratio"])

    else:
        print("No valid mask rule found. Performing dense fine-tuning.")
        # If no valid rule is found, the mask_dict remains empty.
        # The optimizer will then behave like a standard SGD as no mask will be applied.
        pass

    # Convert the mask dictionary into a list of masks for the optimizer
    mask_list = mask_to_param_list(mask_dict, model)

    # === 4. REPLACE THE OPTIMIZER ===
    # Use SparseSGDM instead of the standard SGD, passing the created mask.
    optimizer = SparseSGDM(
        model.parameters(),
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=cfg.get('weight_decay', 0.0),
        mask=mask_list # Pass the computed mask to the optimizer
    )


    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    start_epoch, logger = resume_if_possible(cfg, model, optimizer, scheduler)
    
    best_val_acc = 0.0
    patience = cfg.get("early_stopping_patience", 5)  # valore di default
    patience_counter = 0

    # Main training loop
    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate on the validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step() # Update learning rate

        # Print and log training and validation metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        logger.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        

        # === Early stopping ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # reset patience
        else:
            patience_counter += 1
            print(f"🕓 Early stopping patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\n⛔ Early stopping activated at {epoch+1} (val_acc hasn't improved for {patience} epochs)")
            break

        # === Checkpoint standard ===
        os.makedirs(os.path.dirname(cfg['checkpoint_path']), exist_ok=True)
        os.makedirs(os.path.dirname(cfg['log_path']), exist_ok=True)
        save_checkpoint(model, optimizer, scheduler, epoch + 1, path=cfg['checkpoint_path'])

        os.makedirs(os.path.dirname(cfg['checkpoint_drive_path']), exist_ok=True)
        shutil.copy(cfg['checkpoint_path'], cfg['checkpoint_drive_path'])

        if "log_drive_path" in cfg:
            os.makedirs(os.path.dirname(cfg["log_drive_path"]), exist_ok=True)
            if os.path.exists(cfg['log_path']):
                shutil.copy(cfg['log_path'], cfg["log_drive_path"])
                print(f"Log locale: {cfg['log_path']}")
                print(f"Log Drive: {cfg['log_drive_path']}")

        print(f"Checkpoint locale salvato: {cfg['checkpoint_path']}")
        print(f"Checkpoint backup Drive: {cfg['checkpoint_drive_path']}")

    # Final evaluation on the test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n✅ Final Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
