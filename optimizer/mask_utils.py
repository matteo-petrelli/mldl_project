import torch

def compute_fisher_diagonal(model, dataloader, criterion, device):
    """
    Approximates the Fisher Information Matrix diagonals.
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize a dictionary to store the Fisher information for each parameter
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    
    # Iterate over the dataset
    for inputs, labels in dataloader:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)
        # Reset gradients
        model.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backward pass to compute gradients
        loss.backward()

        # Accumulate the squared gradients for each parameter
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
                
    # Average the Fisher information over the number of batches
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

    # Only sample if the total is significantly larger
    if total_elements > sample_size * 2: 
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

        # Calculate the threshold based on the sampled scores
        threshold = torch.quantile(sampled_all_scores, sparsity_ratio)
        print(f"Warning: Quantile calculated on a sample of {sampled_all_scores.numel()} elements due to large tensor size.")
    else:
        # If the total size is manageable, concatenate all scores
        all_scores = torch.cat(all_scores_list)
        # Calculate the threshold based on all scores
        threshold = torch.quantile(all_scores, sparsity_ratio)

    # Create the mask based on the threshold
    mask = {}
    for name, scores in fisher_dict.items():
        if pick_least_sensitive:
            # Mask weights with scores below the threshold (less important)
            mask[name] = (scores < threshold).float()
        else:
            # Mask weights with scores at or above the threshold (more important)
            mask[name] = (scores >= threshold).float()

    return mask


# Insert this updated function into optimizer/mask_utils.py

def build_mask_by_magnitude(model, sparsity_ratio: float, pick_highest_magnitude: bool = True) -> dict:
    """
    Creates a binary mask based on the absolute magnitude of model weights.
    Includes sampling for large models to prevent memory errors.
    """
    # 1. Collect the magnitudes in a list instead of concatenating them immediately
    all_magnitudes_list = [p.data.abs().view(-1) for n, p in model.named_parameters() if p.requires_grad]

    # 2. Implement the sampling logic (copied from build_mask_by_sensitivity)
    sample_size = 1_000_000
    total_elements = sum(p.numel() for p in all_magnitudes_list)

    if total_elements > sample_size * 2:
        # If the model is too large, calculate the quantile on a sample
        sampled_elements = []
        # For more robust sampling, one could permute each tensor,
        # but for simplicity, we take elements proportionally.
        for magnitudes in all_magnitudes_list:
            num_to_sample = int(round((magnitudes.numel() / total_elements) * sample_size))
            if num_to_sample > 0:
                perm = torch.randperm(magnitudes.numel(), device=magnitudes.device)
                sampled_elements.append(magnitudes.view(-1)[perm[:num_to_sample]])
        
        if sampled_elements:
            sampled_all_magnitudes = torch.cat(sampled_elements)
            threshold = torch.quantile(sampled_all_magnitudes, sparsity_ratio)
            print(f"Warning: Magnitude quantile calculated on a sample of {sampled_all_magnitudes.numel()} elements.")
        else:
            # Fallback in the unlikely event that nothing is sampled
            all_magnitudes = torch.cat(all_magnitudes_list)
            threshold = torch.quantile(all_magnitudes, sparsity_ratio)

    else:
        # If the model is small enough, use the original method
        all_magnitudes = torch.cat(all_magnitudes_list)
        threshold = torch.quantile(all_magnitudes, sparsity_ratio)

    # 3. The logic for creating the mask remains unchanged
    mask = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if pick_highest_magnitude:
                # Mask weights with absolute magnitude at or above the threshold
                mask[name] = (p.data.abs() >= threshold).float()
            else:
                # Mask weights with absolute magnitude below the threshold
                mask[name] = (p.data.abs() < threshold).float()
        else:
            # For parameters that do not require gradients, keep them all (mask of ones)
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
            # For parameters that do not require gradients, keep them all (mask of ones)
            mask[name] = torch.ones_like(p.data)
    return mask
