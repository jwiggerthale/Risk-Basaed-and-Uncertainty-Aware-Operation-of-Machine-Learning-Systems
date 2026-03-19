'''
This script implements model for diabetes dataset with dropout
'''

import torch.nn as nn
import torch



'''
class which implements feed forward NN with dropout for tabular data
'''
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
            nn.Dropout(p=0.2)
            self.layers.append(nn.ReLU())
            last_dim = layer
        self.mu = nn.Sequential(
            nn.Linear(last_dim, num_classes)
        )
        self.log_var = nn.Sequential(
            nn.Linear(last_dim, num_classes)  
            )
        # WICHTIG: Kein Sigmoid hier!
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.tensor, 
               inference: bool = False):
        features = self.model(x)
        mu = self.mu(features)
        log_var = self.log_var(features)
        return mu, log_var
