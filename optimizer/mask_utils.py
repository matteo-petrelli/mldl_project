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


# Inserisci questa funzione aggiornata in optimizer/mask_utils.py

def build_mask_by_magnitude(model, sparsity_ratio: float, pick_highest_magnitude: bool = True) -> dict:
    """
    Creates a binary mask based on the absolute magnitude of model weights.
    Includes sampling for large models to prevent memory errors.
    """
    # 1. Raccogli le magnitudo in una lista invece di concatenarle subito
    all_magnitudes_list = [p.data.abs().view(-1) for n, p in model.named_parameters() if p.requires_grad]

    # 2. Implementa la logica di campionamento (copiata da build_mask_by_sensitivity)
    sample_size = 1_000_000
    total_elements = sum(p.numel() for p in all_magnitudes_list)

    if total_elements > sample_size * 2:
        # Se il modello è troppo grande, calcola il quantile su un campione
        sampled_elements = []
        # Per un campionamento più robusto, si potrebbe permutare ogni tensore,
        # ma per semplicità prendiamo elementi proporzionalmente.
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
            # Fallback nel caso improbabile che non venga campionato nulla
            all_magnitudes = torch.cat(all_magnitudes_list)
            threshold = torch.quantile(all_magnitudes, sparsity_ratio)

    else:
        # Se il modello è abbastanza piccolo, usa il metodo originale
        all_magnitudes = torch.cat(all_magnitudes_list)
        threshold = torch.quantile(all_magnitudes, sparsity_ratio)

    # 3. La logica per creare la maschera rimane invariata
    mask = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if pick_highest_magnitude:
                mask[name] = (p.data.abs() >= threshold).float()
            else:
                mask[name] = (p.data.abs() < threshold).float()
        else:
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

