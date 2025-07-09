import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import json
from tqdm import tqdm
import shutil

# --- Custom Module Imports ---
from data.cifar100_loader import get_transforms, load_cifar100
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import MetricLogger
from models.vit_dino import get_dino_vit_s16

def load_vit_dino_backbone(num_classes):
    """Loads the DINO ViT model and freezes the backbone for linear probing."""
    model = get_dino_vit_s16(num_classes)
    # Freeze all parameters in the model initially
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze only the parameters of the final classification head
    for param in model.head.parameters():
        param.requires_grad = True
    return model
    
def resume_if_possible(cfg, model, optimizer, scheduler):
    """Resumes training from the latest checkpoint and log file."""
    log_path = cfg.get('log_drive_path', cfg['log_path']) # Prioritize Drive log
    logger = MetricLogger(save_path=log_path)
    start_epoch = 0

    # Attempt to resume logs
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                logger.metrics = json.load(f)
            start_epoch = len(logger.metrics)
            print(f"[Logger] Resumed log from epoch {start_epoch}")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous logs: {e}")

    # Find the latest checkpoint (prioritizing Drive) and load it
    resume_path = None
    if os.path.exists(cfg.get("checkpoint_drive_path", "")):
        resume_path = cfg["checkpoint_drive_path"]
    elif os.path.exists(cfg.get("checkpoint_path", "")):
        resume_path = cfg["checkpoint_path"]

    if resume_path:
        checkpoint_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = max(start_epoch, checkpoint_epoch)
        print(f"[Checkpoint] Resumed from epoch {checkpoint_epoch} ({resume_path})")

    return start_epoch, logger

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one training epoch, returning the average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass and loss calculation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
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
        for images, labels in tqdm(dataloader, desc="Validation"):
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
    # --- Setup ---
    # Load config, set device, and prepare data
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, test_tf = get_transforms()
    trainset, valset, testset = load_cifar100(train_tf, test_tf)
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # --- Model, Optimizer, and Scheduler ---
    # Load model for linear probing (only the head is trained)
    model = load_vit_dino_backbone(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer will only act on parameters with requires_grad=True (the head)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=cfg['weight_decay']
    )
    
    # Configure the learning rate scheduler
    if cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    elif cfg.get("scheduler") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 30), gamma=cfg.get("gamma", 0.1))
    else:
        scheduler = None

    # --- Training Initialization ---
    start_epoch, logger = resume_if_possible(cfg, model, optimizer, scheduler)
    best_val_acc = 0.0
    patience = cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    # --- Main Training Loop ---
    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step()

        # Log metrics for the epoch
        logger.log({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # --- Early Stopping Logic ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping activated at epoch {epoch+1}")
            break

        # --- Checkpointing ---
        # Save checkpoint locally and then copy to a backup location (e.g., Google Drive)
        save_checkpoint(model, optimizer, scheduler, epoch + 1, path=cfg['checkpoint_path'])
        shutil.copy(cfg['checkpoint_path'], cfg['checkpoint_drive_path'])
        print(f"Local checkpoint saved: {cfg['checkpoint_path']}")
        print(f"Drive backup checkpoint: {cfg['checkpoint_drive_path']}")

    # --- Final Evaluation ---
    print("\nTraining finished. Evaluating on the test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    # Script entry point: parses the config file argument and starts training
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    main(args)
