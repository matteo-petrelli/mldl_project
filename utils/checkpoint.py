import torch
import os

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Saves model, optimizer and scheduler state.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Loads model and optionally optimizer/scheduler states.
    Returns the epoch to resume from.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0)
