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

import os
import json

def resume_if_possible(cfg, model):
    """
    Resume training from checkpoint and logs. Prefer Drive paths if available.
    """
    local_log_path = cfg['log_path']
    #Ensure the local log directory exists
    os.makedirs(os.path.dirname(local_log_path), exist_ok=True)

    # If the local file doesn't exist, initialize it as empty
    if not os.path.exists(local_log_path):
        with open(local_log_path, 'w') as f:
            json.dump([], f)

    # Initialize logger with the local path
    logger = MetricLogger(save_path=local_log_path)
    start_round = 1

    # Resume logger if the local file is valid
    if os.path.exists(local_log_path) and os.path.getsize(local_log_path) > 0:
        try:
            with open(local_log_path, 'r') as f:
                prev_metrics = json.load(f)
            logger.metrics = prev_metrics
            start_round = len(prev_metrics) + 1
            print(f"[Logger] Resumed from round {start_round} (local: {local_log_path})")
        except Exception as e:
            print(f"[Logger Warning] Failed to load previous local logs: {e}")

    # Resume model checkpoint (Drive has priority for loading checkpoint)
    resume_path = None
    if cfg.get("checkpoint_drive_path") and os.path.exists(cfg["checkpoint_drive_path"]):
        resume_path = cfg["checkpoint_drive_path"]
    elif cfg.get("checkpoint_path") and os.path.exists(cfg["checkpoint_path"]):
        resume_path = cfg["checkpoint_path"]

    if resume_path:
        try:
            checkpoint_round = load_checkpoint(resume_path, model, optimizer=None, scheduler=None)
            start_round = max(start_round, checkpoint_round + 1)
            print(f"[Checkpoint] Resumed from round {checkpoint_round} ({resume_path})")
        except Exception as e:
            print(f"[Checkpoint Warning] Failed to load checkpoint: {e}")

    return start_round, logger

# La funzione train_local ora deve accettare la configurazione per creare la maschera
def train_local_sparse(model, dataloader, criterion, device, cfg):
    """
    Esegue il training locale sparso per un singolo client.
    """
    # 1. Calibra la maschera specifica per questo client sui suoi dati locali
    mask_rule = cfg.get("mask_calibration_rule")
    mask_dict = {}

    if "sensitivity" in mask_rule:
        fisher = compute_fisher_diagonal(model, dataloader, criterion, device)
        if mask_rule == "sensitivity_least":
            mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=True)
        else: # sensitivity_most
            mask_dict = build_mask_by_sensitivity(fisher, cfg["sparsity_ratio"], pick_least_sensitive=False)
    
    elif "magnitude" in mask_rule:
        if mask_rule == "magnitude_lowest":
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=False)
        else: # magnitude_highest
            mask_dict = build_mask_by_magnitude(model, cfg["sparsity_ratio"], pick_highest_magnitude=True)

    elif mask_rule == "random":
        mask_dict = build_mask_randomly(model, cfg["sparsity_ratio"])

    mask_list = mask_to_param_list(mask_dict, model)
    
    # 2. Inizializza l'optimizer sparso
    optimizer = SparseSGDM(model.parameters(), lr=cfg["lr"], momentum=0.9, mask=mask_list)
    
    # 3. Training locale
    model.train()
    for _ in range(cfg["J"]): # J = epoche locali
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, test_tf = get_transforms()
    trainset, _, testset = load_cifar100(train_tf, test_tf)
    test_loader = DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False)

    # Per gestire IID e non-IID, usa lo sharding dal config
    if cfg["sharding"] == "iid":
        client_datasets = iid_split(trainset, cfg["K"])
    else:
        client_datasets = noniid_split(trainset, cfg["K"], cfg["Nc"])

    global_model = get_dino_vit_s16(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    for round_num in range(1, cfg["rounds"] + 1):
        print(f"\n--- Round {round_num} ---")
        local_models = []
        selected_clients = torch.randperm(cfg["K"])[:int(cfg["K"] * cfg["C"])]

        for client_id in tqdm(selected_clients, desc="Clients training"):
            client_model = deepcopy(global_model)
            client_model.to(device)
            client_loader = DataLoader(client_datasets[client_id], batch_size=cfg["batch_size"], shuffle=True)
            
            # Esegui il training locale SPARSO
            local_state = train_local_sparse(client_model, client_loader, criterion, device, cfg)
            local_models.append(local_state)
            
        # Aggrega i modelli come prima
        global_model = aggregate_models(global_model, local_models)

        # Valuta il modello globale
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        logger.log({
            "round": round_num,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        if round_num % cfg["save_every"] == 0:
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
