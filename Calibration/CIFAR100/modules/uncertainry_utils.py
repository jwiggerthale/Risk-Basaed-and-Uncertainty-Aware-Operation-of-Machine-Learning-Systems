#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:03:12 2025

@author: jwiggerthale
"""

'''
This file implements different function for uncertainty estimations
  - heteroscedastic loss funciton used to train models
  - mc dropout forward pass
'''

import torch
import torch.nn.functional as F


'''
function which implements heteroscedastische cross-entropy via logit-sampling (following gal 2017)
call with:
    mu:       (B, C) mean logits (predicted by model)
    log_var:  (B, C) log_var (predicted by model)
    target:   (B,)   class labels
    num_samples: number of samples
    penalty:  regularizer 
    
'''
def heteroscedastic_ce(mu, 
                       log_var, 
                       target, 
                       num_samples=10, 
                       reduction="mean", 
                       var_epsilon=1e-6, 
                       penalty=1e-6):
    var = F.softplus(log_var) + var_epsilon # positive values; softplus for stability
    std = torch.sqrt(var)

    # z = mu + std * eps, eps ~ N(0, I)
    B, C = mu.shape
    eps = torch.randn(num_samples, B, C, device=mu.device, dtype=mu.dtype)
    logits_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (S, B, C)

    # cross entropy per sample
    ce = []
    for s in range(num_samples):
        ce_s = F.cross_entropy(logits_samples[s], target, reduction="none")  # (B,)
        ce.append(ce_s)
    ce = torch.stack(ce, dim=0).mean(dim=0)  
    # penalty avoids unlimited growth of variance
    var_pen = penalty * var.mean(dim=1)  # (B,)
    loss = ce + var_pen
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # (B,)
      

'''
function to perform forward pass with MC dropout and heteroscedastic component
call with:
  model: nn.Module --> model to use
  x: torch.tensor --> sample to make prediction on 
  n_dropout --> number of mc samples 
  n_noise --> number of samples for heteroscedastic loss (AU)
returns: 
  dict with:
        "probs": mean predicted value (across mc samples)
        "pred_entropy" --> prediction entropy
        "au" -->    mean AU
        "eu" -->   eu (STD from probs across mc samples)
        "probs_var_across_dropout" --> variance across dropout 
'''
@torch.no_grad()
def mc_dropout_with_heteroscedastic(model, x, n_dropout=30, n_noise=10):
    """
    Speichert:
      - probs_mean_d: Mittel Ã¼ber Logit-Samples je Dropout-Pass (D, B, C)
      - Daraus: H[E_{w,z}[p]], E_{w,z}[H[p]], und BALD.
    """
    import torch.nn.functional as F

    def entropy(p, dim=-1, eps=1e-12):
        p = p.clamp_min(eps)
        return -(p * p.log())#.sum(dim=dim)

    model.train()  # Dropout an
    probs_mean_per_d = []

    exp_entropy_accum = 0.0  # E_{w,z}[H[p]]
    for _ in range(n_dropout):
        pred = model(x)
        if len(pred) == 3:
            mu, logvar, _ = pred
        else:
            mu, logvar = pred                     # (B, C)
        std = torch.exp(0.5 * logvar)
        B, C = mu.shape

        eps = torch.randn(n_noise, B, C, device=mu.device)
        logits = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # (S, B, C)
        probs  = F.softmax(logits, dim=-1)                 # (S, B, C)

        probs_mean_d = probs.mean(dim=0)                   # (B, C)  -> speichere!
        probs_mean_per_d.append(probs_mean_d)

        exp_entropy_accum += entropy(probs, dim=-1).mean(dim=0)  # (B,)

    probs_mean_stack = torch.stack(probs_mean_per_d, dim=0)      # (D, B, C)
    mean_probs = probs_mean_stack.mean(dim=0)                    # (B, C)

    pred_entropy = entropy(mean_probs)                           # H[E_{w,z}[p]]
    expected_entropy = exp_entropy_accum / n_dropout             # E_{w,z}[H[p]]
    epistemic_entropy = (pred_entropy - expected_entropy).clamp_min(0.0)
    eu = probs_mean_stack.std(dim= 0)#.mean(dim = -1)
    # Varianz der gemittelten Wahrscheinlichkeiten zwischen Dropout-PÃ¤ssen
    probs_var_across_dropout = probs_mean_stack.var(dim=0, unbiased=False).mean(dim=-1)  # (B,)

    return {
        "probs": mean_probs,
        "pred_entropy": pred_entropy,
        "au": expected_entropy,     # aleatorischer Anteil im Mittel
        "eu": eu,   # BALD-Anteil
        "probs_var_across_dropout": probs_var_across_dropout,
    }



@torch.no_grad()
def entropy_from_probs(p, dim=-1, eps=1e-12):
    # p: (..., C)
    p = p.clamp_min(eps)
    return -(p * p.log())#.sum(dim=dim)
