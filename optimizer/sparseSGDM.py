import torch
from torch.optim import SGD

class SparseSGDM(SGD):
    """
    An SGD optimizer that supports gradient masking for sparse updates.
    This allows only a subset of gradients to be applied during the update step.
    """

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, mask=None):
        """Initializes the optimizer and stores the sparsity mask."""
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        # The mask is a list of tensors, one for each parameter, where 1 means update and 0 means freeze.
        self.mask = mask

    def step(self, closure=None):
        """Performs a single optimization step, applying the gradient mask first."""
        loss = None
        if closure is not None:
            loss = closure()

        # Apply the mask to the gradients in-place before the update.
        for group in self.param_groups:
            # The mask should be a list of tensors, one for each parameter in the group.
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Multiply the gradient by the corresponding binary mask.
                if self.mask and i < len(self.mask):
                    p.grad.data *= self.mask[i]

        # Call the original SGD step function, which will now use the masked gradients.
        super().step()
        return loss
