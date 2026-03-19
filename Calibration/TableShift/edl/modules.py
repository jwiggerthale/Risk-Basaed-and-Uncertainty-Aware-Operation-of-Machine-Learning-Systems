'''
This script implements an EDL classification model for the diabetes dataset
'''

import torch.nn as nn
import torch
from .edl_basics import Dirichlet


class heart_model(nn.Module):
    def __init__(self, 
                 input_dim: int = 21,
                 layers: list = [512, 256, 256, 128, 128, 64],
                 num_classes: int = 2, 
                 temperature: float = 1.5):
        super().__init__()
        self.layers = []
        last_dim = input_dim
        for layer in layers:
            self.layers.append(nn.Linear(last_dim, layer))
            self.layers.append(nn.ReLU())
            last_dim = layer
        self.layers.append(Dirichlet(last_dim, num_classes))
        self.model = nn.Sequential(*self.layers)
        self.temperature = temperature #nn.Parameter(torch.tensor(temperature))
    def forward(self, x: torch.tensor):
        logits = self.model(x)
        return logits
