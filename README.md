# mldl_project
Federeted Learning Project
This repository contains implementations for centralized and federated learning experiments, including sparse update mechanisms, using a DINO ViT-S/16 backbone on the CIFAR-100 dataset.
Setup
Clone the repository:

git clone https://github.com/your-username/MLDL_PROJECT-MAIN.git
cd MLDL_PROJECT-MAIN

Install dependencies:

pip install -r requirements.txt

How to Run Experiments
Experiments are configured via YAML files located in the configs/ directory.

To run an experiment, use the following command format:

python3 experiments/<experiment_script_name>.py --config configs/<config_file_name>.yaml

Example (for Google Colab after cloning and setting as reference folder):

!python3 experiments/federated_sparse.py --config configs/federated_sparse_nonidd_1.yaml

Replace federated_sparse.py with the desired experiment script and federated_sparse_nonidd_1.yaml with the appropriate configuration file.
