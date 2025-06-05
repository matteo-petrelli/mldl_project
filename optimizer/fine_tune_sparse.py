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
from sparse.mask_utils import compute_fisher_diagonal, build_mask_from_fisher
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

    # Simula 1 client per calcolo Fisher
    client_datasets = noniid_split(trainset, cfg["K"], cfg["Nc"])
    fisher_loader = DataLoader(client_datasets[cfg["client_id"]], batch_size=cfg["batch_size"], shuffle=True)

    model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    # Step 1: Calcolo della Fisher diagonale
    fisher = compute_fisher_diagonal(model, fisher_loader, criterion, device)

    # Step 2: Crea maschera
    mask_dict = build_mask_from_fisher(fisher, cfg["sparsity_ratio"])
    mask_list = mask_to_param_list(mask_dict, model)

    # Step 3: Fine-tuning con maschera
    train_loader = fisher_loader
    optimizer = SparseSGDM(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=0.0005, mask=mask_list)
    logger = MetricLogger(cfg["log_path"])

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
