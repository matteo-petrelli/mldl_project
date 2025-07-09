# MLDL_PROJECT-MAIN

This repository contains code for training Vision Transformers (ViTs) using centralized and federated learning approaches, with support for sparse optimization.

## Folder Structure

- `configs/`: YAML configuration files for different training setups.
- `data/`: Data loading utilities. Includes `cifar100_loader.py` for loading the CIFAR-100 dataset.
- `experiments/`: Contains scripts for running experiments:
  - `centralized_training.py`: Standard centralized training.
  - `centralized_sparse.py`: Centralized training with sparse optimization.
  - `federated_training.py`: Federated training.
  - `federated_sparse.py`: Federated training with sparse optimization.
- `models/`: Model architectures. Includes `vit_dino.py` (Vision Transformer based on DINO).
- `optimizer/`: Sparse optimization utilities:
  - `sparseSGDM.py`: Sparse SGD optimizer.
  - `mask_utils.py`: Functions for creating and managing sparse masks.
- `utils/`: Utility scripts for logging and checkpointing:
  - `logger.py`, `checkpoint.py`.
- `requirements.txt`: Python dependencies.

## Running an Experiment

To run an experiment, clone the repository, install the dependencies, and use a command like the following (example for federated sparse training):

```bash
!python3 experiments/federated_sparse.py --config configs/federated_sparse_nonidd_1.yaml
