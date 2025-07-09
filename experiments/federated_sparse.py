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
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.stack([client_state[key].float() for client_state in local_models], dim=0).mean(dim=0)
    global_model.load_state_dict(global_state)
    return global_model

def evaluate(model, dataloader, criterion, device):
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
    param_masks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in mask_dict:
                param_masks.append(mask_dict[name].to(param.device))
            else:
                param_masks.append(torch.ones_like(param))
    return param_masks

def resume_if_possible(cfg, model):
    local_log_path = cfg['log_path']
    os.makedirs(os.path.dirname(local_log_path), exist_ok=True)
    if not os.path.exists(local_log_path):
        with open(local_log_path, 'w') as f:
            json.dump([], f)
    logger = MetricLogger(save_path=local_log_path)
    start_round = 1
    if os.path.exists(local_log_path) and os.path.getsize(local_log_path) > 0:
        try:
            with open(local_log_path, 'r') as f:
                prev_metrics = json.load(f)
            logger.metrics = prev_metrics
            start_round = len(prev_metrics) + 1
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous local logs: {e}")
    resume_path = None
    if cfg.get("checkpoint_drive_path") and os.path.exists(cfg["checkpoint_drive_path"]):
        resume_path = cfg["checkpoint_drive_path"]
    elif cfg.get("checkpoint_path") and os.path.exists(cfg["checkpoint_path"]):
        resume_path = cfg["checkpoint_path"]
    if resume_path:
        try:
            checkpoint_round = load_checkpoint(resume_path, model, optimizer=None, scheduler=None)
            start_round = max(start_round, checkpoint_round + 1)
        except Exception as e:
            print(f"[Checkpoint Warning] Failed to load checkpoint: {e}")
    return start_round, logger

def train_local_sparse(model, dataloader, criterion, device, cfg, existing_mask=None):
    mask_dict = {}
    if existing_mask is not None:
        mask_dict = existing_mask
    else:
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
    
    mask_list = mask_to_param_list(mask_dict, model)
    optimizer = SparseSGDM(model.parameters(), lr=cfg["lr"], momentum=0.9, mask=mask_list)
    
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
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, test_tf = get_transforms()
    trainset, _, testset = load_cifar100(train_tf, test_tf, val_ratio=0.0)
    test_loader = DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    if cfg["sharding"] == "iid":
        client_datasets = iid_split(trainset, cfg["K"])
    else:
        client_datasets = noniid_split(trainset, cfg["K"], cfg.get("Nc", 10))

    global_model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    start_round, logger = resume_if_possible(cfg, global_model)

    client_masks = {}
    # --- MODIFICA 1: Ripristino della variabile e della logica per la fase iniziale ---
    # Se non specificato, ogni round è di calibrazione (comportamento originale)
    calibration_rounds = cfg.get("calibration_rounds", cfg["rounds"])

    for round_num in range(start_round, cfg["rounds"] + 1):
        # --- MODIFICA 2: La condizione torna ad essere un controllo sulla fase iniziale ---
        is_calibration_round = (round_num <= calibration_rounds)

        print(f"\n--- Round {round_num}/{cfg['rounds']} ---")
        if is_calibration_round:
            print("Mode: 🛠️  INITIAL CALIBRATION PHASE")
        else:
            print("Mode: 💪 FINE-TUNING (using fixed masks)")

        local_models = []
        selected_clients = torch.randperm(cfg["K"])[:int(cfg["K"] * cfg["C"])]

        for client_id_tensor in tqdm(selected_clients, desc="Clients training"):
            client_id = client_id_tensor.item()
            client_model = deepcopy(global_model)
            client_model.to(device)
            client_loader = DataLoader(client_datasets[client_id], batch_size=cfg["batch_size"], shuffle=True)
            
            # La logica di decisione funziona anche con questo scheduling
            mask_to_use = None
            if not is_calibration_round:
                mask_to_use = client_masks.get(client_id, None)
            
            local_state, used_mask = train_local_sparse(client_model, client_loader, criterion, device, cfg, existing_mask=mask_to_use)
            local_models.append(local_state)
            
            # Salva la maschera calcolata/usata per il client. 
            # Dopo la fase di calibrazione, questa operazione semplicemente ri-salva la stessa maschera.
            client_masks[client_id] = used_mask
            
        global_model = aggregate_models(global_model, local_models)
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        logger.log({ "round": round_num, "test_loss": test_loss, "test_acc": test_acc })

        if round_num % cfg.get("save_every", 10) == 0:
            os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
            save_checkpoint(global_model, None, None, round_num, cfg["checkpoint_path"])
            print(f"[Checkpoint] Salvato localmente: {cfg['checkpoint_path']}")
            if "checkpoint_drive_path" in cfg:
                os.makedirs(os.path.dirname(cfg["checkpoint_drive_path"]), exist_ok=True)
                shutil.copy(cfg["checkpoint_path"], cfg["checkpoint_drive_path"])
                print(f"[Checkpoint] Backup su Drive: {cfg['checkpoint_drive_path']}")
            if "log_drive_path" in cfg:
                os.makedirs(os.path.dirname(cfg["log_drive_path"]), exist_ok=True)
                if os.path.exists(cfg["log_path"]):
                    shutil.copy(cfg["log_path"], cfg["log_drive_path"])
                    print(f"[Log] Copiato su Drive: {cfg['log_drive_path']}")
                else:
                    print(f"[Log Warning] Il file di log '{cfg['log_path']}' non esiste e non è stato copiato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
