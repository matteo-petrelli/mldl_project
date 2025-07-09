import torch

def compute_fisher_diagonal(model, dataloader, criterion, device):
    """Approximates the Fisher Information Matrix diagonals using squared gradients."""
    # Initialize a dictionary to store the Fisher information for each parameter.
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()

    # Accumulate the squared gradients over the dataset.
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

    # Average the Fisher information across the number of batches.
    for n in fisher:
        fisher[n] /= len(dataloader)
    return fisher

def build_mask_by_sensitivity(fisher_dict: dict, sparsity_ratio: float, pick_least_sensitive: bool = False) -> dict:
    """Creates a binary mask by thresholding Fisher scores to identify important parameters."""
    all_scores_list = [f.view(-1) for f in fisher_dict.values()]
    total_elements = sum(f.numel() for f in fisher_dict.values())
    sample_size = 1_000_000

    # For very large models, calculate the threshold on a sample to avoid memory errors.
    if total_elements > sample_size * 2:
        sampled_elements = []
        for scores in all_scores_list:
            # Proportionally sample from each parameter tensor.
            num_to_sample = int(round((scores.numel() / total_elements) * sample_size))
            if num_to_sample > 0:
                perm = torch.randperm(scores.numel(), device=scores.device)
                sampled_elements.append(scores.view(-1)[perm[:num_to_sample]])
        
        sampled_all_scores = torch.cat(sampled_elements)
        threshold = torch.quantile(sampled_all_scores, sparsity_ratio)
        print(f"Warning: Quantile calculated on a sample of {sampled_all_scores.numel()} elements.")
    else:
        # If the model is small enough, use all scores to calculate the exact threshold.
        all_scores = torch.cat(all_scores_list)
        threshold = torch.quantile(all_scores, sparsity_ratio)

    # Create a binary mask for each parameter based on the threshold.
    mask = {}
    for name, scores in fisher_dict.items():
        if pick_least_sensitive:
            # Keep parameters with scores *below* the threshold (less sensitive).
            mask[name] = (scores < threshold).float()
        else:
            # Keep parameters with scores *at or above* the threshold (more sensitive).
            mask[name] = (scores >= threshold).float()
    return mask

def build_mask_by_magnitude(model, sparsity_ratio: float, pick_highest_magnitude: bool = True) -> dict:
    """Creates a binary mask based on the absolute magnitude of model weights."""
    # Collect the absolute magnitudes of all trainable parameters.
    all_magnitudes_list = [p.data.abs().view(-1) for n, p in model.named_parameters() if p.requires_grad]
    total_elements = sum(p.numel() for p in all_magnitudes_list)
    sample_size = 1_000_000

    # For large models, calculate the threshold on a sample to save memory.
    if total_elements > sample_size * 2:
        sampled_elements = []
        for magnitudes in all_magnitudes_list:
            num_to_sample = int(round((magnitudes.numel() / total_elements) * sample_size))
            if num_to_sample > 0:
                perm = torch.randperm(magnitudes.numel(), device=magnitudes.device)
                sampled_elements.append(magnitudes.view(-1)[perm[:num_to_sample]])
        
        sampled_all_magnitudes = torch.cat(sampled_elements)
        threshold = torch.quantile(sampled_all_magnitudes, sparsity_ratio)
        print(f"Warning: Magnitude quantile calculated on a sample of {sampled_all_magnitudes.numel()} elements.")
    else:
        # If the model is small, use all weights for an exact threshold.
        all_magnitudes = torch.cat(all_magnitudes_list)
        threshold = torch.quantile(all_magnitudes, sparsity_ratio)

    # Create the binary mask based on the calculated threshold.
    mask = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if pick_highest_magnitude:
                # Keep parameters with magnitudes at or above the threshold.
                mask[name] = (p.data.abs() >= threshold).float()
            else:
                # Keep parameters with magnitudes below the threshold.
                mask[name] = (p.data.abs() < threshold).float()
    return mask
    
def build_mask_randomly(model, sparsity_ratio: float) -> dict:
    """Creates a binary mask by randomly selecting a fraction of parameters to keep."""
    mask = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            # A value of 1 means the parameter is kept (updated), 0 means it's pruned.
            # `torch.rand_like > ratio` results in approx. (1-ratio) of parameters being kept.
            mask[name] = (torch.rand_like(p.data) > sparsity_ratio).float()
    return mask
