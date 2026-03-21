'''
This script implements a cost sensitive heteroscedastic loss function
'''

import torch
import torch.nn.functional as F


"""
function which implements heteroscedastic, cost sensitive loss
call with:
    mu: torch.Tensor --> mean logits
    log_var: torch.Tensor --> variance of logits
    target: torch.Tensor --> labels
    cost_mat: torch.Tensor --> cost structure for classification problem
    num_samples: int --> number of samples 
    reduction: str = 'mean' --> how to process results
    var_epsilon: float --> ensure, var is > 0
    penalty: float --> regularizer
    device: str = 'cuda' --> where to calculate
"""
def cost_sensitive_heteroscedastic_ce(mu: torch.tensor, 
                       log_var: torch.tensor, 
                       target: torch.tensor, 
                       cost_mat: torch.tensor,
                       num_samples: int =10, 
                       reduction: str ="mean", 
                       var_epsilon: float =1e-6, 
                       penalty: float =1e-6, 
                       device: str = 'cuda'):
    var = F.softplus(log_var) + var_epsilon
    std = torch.sqrt(var)

    #  z = mu + std * eps, eps ~ N(0, I)
    B, C = mu.shape
    eps = torch.randn(num_samples, B, C, device=mu.device, dtype=mu.dtype)
    logits_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (S, B, C)

    # cross entropy per sample
    ce = []
    for s in range(num_samples):
        ce_s = F.cross_entropy(logits_samples[s], target, reduction="none")  # (B,)
        pred_cls = logits_samples[s].argmax(dim = 1)
        costs = [cost_mat[elem.item()][pred_cls[i].item()] for i, elem in enumerate(target)]
        costs = torch.tensor(costs).to(device)
        ce_s *= costs
        ce.append(ce_s)
    ce = torch.stack(ce, dim=0).mean(dim=0)  # (B,)

    # avoid uncontrolled loss increase
    var_pen = penalty * var.mean(dim=1)  # (B,)

    loss = ce + var_pen
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # (B,)
