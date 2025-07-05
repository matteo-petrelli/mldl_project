import torch

def compute_fisher_diagonal(model, dataloader, criterion, device):
    """
    Approximates the Fisher Information Matrix diagonals.
    """
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
    for n in fisher:
        fisher[n] /= len(dataloader)
    return fisher

def build_mask_by_sensitivity(fisher_dict: dict, sparsity_ratio: float, pick_least_sensitive: bool = False) -> dict:
    """
    Creates a binary mask based on Fisher scores.
    If pick_least_sensitive is True, it selects (mask=1) the least sensitive parameters for update.
    Otherwise (default), it selects (mask=1) the most sensitive parameters for update.

    Args:
        fisher_dict (dict): Dictionary with Fisher scores per parameter.
        sparsity_ratio (float): Percentage of parameters to NOT update (mask to 0).
        pick_least_sensitive (bool): If True, the LEAST sensitive parameters are SELECTED for update.
                                     If False, the MOST sensitive parameters are SELECTED for update.
    Returns:
        dict: Mask dictionary per parameter.
    """
    all_scores = torch.cat([f.view(-1) for f in fisher_dict.values()])

    # Calculate the threshold based on the percentage of parameters to be masked.
    # If sparsity_ratio is 0.8, the threshold will be at the 20th percentile of scores
    # (meaning 80% of scores are below it).
    threshold = torch.quantile(all_scores, sparsity_ratio)
    mask = {}

    for name, scores in fisher_dict.items():
        if pick_least_sensitive:
            # Select the least sensitive for update (score < threshold)
            mask[name] = (scores < threshold).float()
        else:
            # Select the most sensitive for update (score >= threshold)
            mask[name] = (scores >= threshold).float()
            
    return mask


def build_mask_by_magnitude(model, sparsity_ratio: float, pick_highest_magnitude: bool = True) -> dict:
    """
    Creates a binary mask based on the absolute magnitude of model weights.
    If pick_highest_magnitude is True, it selects (mask=1) parameters with higher magnitude for update.
    Otherwise, it selects (mask=1) parameters with lower magnitude for update.

    Args:
        model (torch.nn.Module): The PyTorch model.
        sparsity_ratio (float): Percentage of parameters to NOT update (mask to 0).
        pick_highest_magnitude (bool): If True, parameters with HIGHEST magnitude are SELECTED.
                                       If False, parameters with LOWEST magnitude are SELECTED.
    Returns:
        dict: Mask dictionary per parameter.
    """
    all_magnitudes = torch.cat([p.data.abs().view(-1) for n, p in model.named_parameters() if p.requires_grad])

    # Calculate the threshold
    threshold = torch.quantile(all_magnitudes, sparsity_ratio)

    mask = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if pick_highest_magnitude:
                # Select parameters with HIGHER magnitude (>= threshold)
                mask[name] = (p.data.abs() >= threshold).float()
            else:
                # Select parameters with LOWER magnitude (< threshold)
                mask[name] = (p.data.abs() < threshold).float()
        else:
            # If the parameter does not require gradients, it should not be masked/updated.
            # We set its mask to 1 by default (it won't be touched by the optimizer if requires_grad=False)
            mask[name] = torch.ones_like(p.data)
    return mask

def build_mask_randomly(model, sparsity_ratio: float) -> dict:
    """
    Creates a binary mask by randomly selecting parameters.

    Args:
        model (torch.nn.Module): The PyTorch model.
        sparsity_ratio (float): Percentage of parameters to NOT update (mask to 0).
                                (1 - sparsity_ratio) will be the percentage of randomly updated parameters.
    Returns:
        dict: Mask dictionary per parameter.
    """
    mask = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            # Create a random mask: True to update, False to mask
            # torch.rand_like(p.data) > sparsity_ratio means roughly (1 - sparsity_ratio) of values will be True
            mask[name] = (torch.rand_like(p.data) > sparsity_ratio).float()
        else:
            mask[name] = torch.ones_like(p.data)
    return mask

