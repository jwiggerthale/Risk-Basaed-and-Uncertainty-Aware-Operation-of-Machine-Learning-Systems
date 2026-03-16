'''
This script implements setting the seed to a defined value
'''

import torch
import numpy as np
import random


# ----- Repro -----
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
