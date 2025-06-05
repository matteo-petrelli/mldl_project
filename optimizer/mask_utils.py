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

def build_mask_from_fisher(fisher_dict, sparsity_ratio):
    """
    Create a binary mask keeping only the top (1-sparsity_ratio) sensitive parameters.
    """
    all_scores = torch.cat([f.view(-1) for f in fisher_dict.values()])
    threshold = torch.quantile(all_scores, sparsity_ratio)
    mask = {}

    for name, scores in fisher_dict.items():
        mask[name] = (scores >= threshold).float()
    return mask
