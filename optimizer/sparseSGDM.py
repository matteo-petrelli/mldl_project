import torch
from torch.optim import SGD

class SparseSGDM(SGD):
    """
    SGD optimizer that supports gradient masking (sparse updates).
    """

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, mask=None):
        super(SparseSGDM, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.mask = mask

    def step(self, closure=None):
        """
        Perform a single optimization step with gradient masking.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if self.mask is not None:
                    if isinstance(self.mask, list):
                        d_p *= self.mask[i]
                    else:
                        d_p *= self.mask  # full mask for this param group

        super().step()
        return loss
