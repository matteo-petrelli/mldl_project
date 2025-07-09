import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

# Import project-specific modules
from data.cifar100_loader import get_transforms, load_cifar100
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import MetricLogger
from models.vit_dino import get_dino_vit_s16 

def generate_and_save_plot(metrics, config_filename):
    """
    Generates and saves a plot with training and validation results.
    The plot is saved in a 'plots' directory created in the current folder.
    """
    # Check if there is data to plot
    if not metrics:
        print("Warning: No metrics data to plot.")
        return

    # --- Automatic Path and Filename Generation ---
    # Define the output directory for plots
    plots_dir = "plots"
    # Create the directory if it does not exist
    os.makedirs(plots_dir, exist_ok=True)

    # Generate a filename from the config file name
    base_name = os.path.basename(config_filename)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(plots_dir, f"{name_without_ext}.png")
    # --- End of Automatic Path Logic ---

    # Extract data for plotting
    epochs = [m.get('epoch', i + 1) for i, m in enumerate(metrics)]
    train_acc = [m['train_acc'] for m in metrics]
    val_acc = [m['val_acc'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Training Results for: {name_without_ext}", fontsize=16, weight='bold')

    # Plot Accuracy
    ax1.plot(epochs, train_acc, 'o-', label='Train Accuracy')
    ax1.plot(epochs, val_acc, 'o-', label='Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Trends', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Loss
    ax2.plot(epochs, train_loss, 'o--', label='Train Loss')
    ax2.plot(epochs, val_loss, 'o--', label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Trends', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f" Plot successfully saved to: {output_path}")
    plt.close()


def load_vit_dino_backbone(num_classes):
    """Loads the DINO ViT model and freezes all layers except the final classification head."""
    model = get_dino_vit_s16(num_classes)
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the parameters of the head for training
    for param in model.head.parameters():
        param.requires_grad = True
    return model
    
def resume_if_possible(cfg, model, optimizer, scheduler):
    """Resumes training from a checkpoint and log file if they exist."""
    # Prefer log from Drive if it exists
    log_path = cfg.get('log_drive_path', cfg['log_path'])
    logger = MetricLogger(save_path=log_path)
    start_epoch = 0

    # Try to resume logs
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                prev_metrics = json.load(f)
            logger.metrics = prev_metrics
            start_epoch = len(prev_metrics)
            print(f"[Logger] Resumed log from epoch {start_epoch}")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous logs: {e}")

    # Checkpoint resume: prefer Drive path if available
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
    """Runs a single epoch of training."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
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

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model's performance on the validation set."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def main(args):
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_tf, test_tf = get_transforms()
    trainset, valset, testset = load_cifar100(train_tf, test_tf)

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # Initialize model, criterion, optimizer, and scheduler
    model = load_vit_dino_backbone(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer only considers parameters that require gradients
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=cfg['weight_decay']
    )
    
    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    elif cfg["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 30), gamma=cfg.get("gamma", 0.1))
    else:
        scheduler = None

    # Resume training if a checkpoint exists
    start_epoch, logger = resume_if_possible(cfg, model, optimizer, scheduler)

    best_val_acc = 0.0
    patience = cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    # Main training loop
    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        # Log metrics for the epoch
        logger.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
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

        # === Standard checkpointing ===
        os.makedirs(os.path.dirname(cfg['checkpoint_path']), exist_ok=True)
        os.makedirs(os.path.dirname(cfg['log_path']), exist_ok=True)
        save_checkpoint(model, optimizer, scheduler, epoch + 1, path=cfg['checkpoint_path'])

        # Backup checkpoint and logs to a secondary location (e.g., Google Drive)
        os.makedirs(os.path.dirname(cfg['checkpoint_drive_path']), exist_ok=True)
        shutil.copy(cfg['checkpoint_path'], cfg['checkpoint_drive_path'])

        if "log_drive_path" in cfg:
            os.makedirs(os.path.dirname(cfg["log_drive_path"]), exist_ok=True)
            if os.path.exists(cfg['log_path']):
                shutil.copy(cfg['log_path'], cfg["log_drive_path"])
                print(f"Local log: {cfg['log_path']}")
                print(f"Drive log: {cfg['log_drive_path']}")

        print(f"Local checkpoint saved: {cfg['checkpoint_path']}")
        print(f"Drive backup checkpoint: {cfg['checkpoint_drive_path']}")

    # Final evaluation on the test set after training is complete
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    generate_and_save_plot(logger.get_all(), args.config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
