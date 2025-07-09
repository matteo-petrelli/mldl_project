# MLDL_PROJECT-MAIN

This repository provides code for training Vision Transformers (ViTs) using both centralized and federated learning approaches, with support for sparse fine-tuning using various mask calibration strategies.

## Folder Structure

- `configs/`: YAML configuration files for setting up different experiments.
- `data/`: Dataset loader scripts.
  - `cifar100_loader.py`: Loads the CIFAR-100 dataset.
- `experiments/`: Main training scripts:
  - `centralized_training.py`: Standard centralized training.
  - `centralized_sparse.py`: Centralized training with sparse fine-tuning.
  - `federated_training.py`: Federated learning based on the FedAvg algorithm.
  - `federated_sparse.py`: Federated sparse fine-tuning using FedAvg and different mask calibration rules.
- `models/`: Model definitions.
  - `vit_dino.py`: Vision Transformer model initialized from DINO pretraining.
- `optimizer/`: Sparse optimization utilities.
  - `sparseSGDM.py`: Sparse SGD optimizer implementation.
  - `mask_utils.py`: Utilities for generating and applying sparse masks.
- `utils/`: Support utilities.
  - `logger.py`: Logging utilities.
  - `checkpoint.py`: Checkpointing and saving model states.
- `requirements.txt`: List of required Python packages.

## Setup

Clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd MLDL_PROJECT-MAIN
pip install -r requirements.txt

## Running an Experiment

To run an experiment, clone the repository, install the dependencies, and use a command like the following (example for federated sparse training):

```bash
!python3 experiments/federated_sparse.py --config configs/federated_sparse_nonidd_1.yaml
