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
    ...
    """
    # Collect all scores in a list first, without immediate concatenation
    all_scores_list = [f.view(-1) for f in fisher_dict.values()]

    # Option 1: Sample a subset for quantile calculation if the full tensor is too large
    # Determine a reasonable sample size, e.g., 1 million elements
    sample_size = 1_000_000
    total_elements = sum(f.numel() for f in fisher_dict.values())

    if total_elements > sample_size * 2: # Only sample if the total is significantly larger
        # Concatenate a subset of elements. This is an approximation.
        # A more robust sampling would involve randomly picking elements across all tensors.
        # For simplicity, we'll just take the first 'sample_size' elements
        # from a concatenated view.
        # However, for true random sampling, you'd need to consider how to
        # sample across multiple tensors efficiently without first concatenating all.
        # A simpler way for a large number of parameters would be to iterate
        # and collect a sample.

        # More robust (but potentially slower for many small tensors) way to sample:
        sampled_elements = []
        current_count = 0
        for scores in all_scores_list:
            if current_count >= sample_size:
                break
            num_to_take = min(scores.numel(), sample_size - current_count)
            # Randomly permute and take the first num_to_take
            perm = torch.randperm(scores.numel(), device=scores.device)
            sampled_elements.append(scores.view(-1)[perm[:num_to_take]])
            current_count += num_to_take

        if sampled_elements:
            sampled_all_scores = torch.cat(sampled_elements)
        else:
            raise ValueError("No elements sampled for quantile calculation. Fisher dict might be empty.")

        threshold = torch.quantile(sampled_all_scores, sparsity_ratio)
        print(f"Warning: Quantile calculated on a sample of {sampled_all_scores.numel()} elements due to large tensor size.")
    else:
        # If the total size is manageable, concatenate all scores
        all_scores = torch.cat(all_scores_list)
        threshold = torch.quantile(all_scores, sparsity_ratio)

    mask = {}
    for name, scores in fisher_dict.items():
        if pick_least_sensitive:
            mask[name] = (scores < threshold).float()
        else:
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

