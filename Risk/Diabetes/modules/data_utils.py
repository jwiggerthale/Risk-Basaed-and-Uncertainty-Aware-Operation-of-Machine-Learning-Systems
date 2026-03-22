'''
This script implements a simple torch data loader from tabular data
'''

from torch.utils.data import Dataset



class heart_ds(Dataset):
    def __init__(self, 
                 x: list, 
                 y: list):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, 
                  idx: int):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
