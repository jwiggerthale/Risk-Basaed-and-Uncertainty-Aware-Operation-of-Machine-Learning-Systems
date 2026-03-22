'''
This script implements a feed forward NN
'''

import torch.nn as nn
import torch



'''
Feed forward NN
(called herat model since I first implemented the model on a heard disease prediction dataset)
initialize with: 
    input_dim: int = 21,--> how many features are in your dataset
    layers: list = [512, 256, 256, 128, 128, 64], --> number of neurons in hidden layers
    num_classes: int = 2, --> number of output neurons (classes in the dataset)
    temperature: float = 1.5 --> temperature factor for temperature scaling; experimental; not used for thesis (set to 1)
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
        self.temperature = temperature #nn.Parameter(torch.tensor(temperature))
    
    def forward(self, x: torch.tensor):
        features = self.model(x)
        mu = self.mu(features)/self.temperature
        log_var = self.log_var(features)
        return mu, log_var
