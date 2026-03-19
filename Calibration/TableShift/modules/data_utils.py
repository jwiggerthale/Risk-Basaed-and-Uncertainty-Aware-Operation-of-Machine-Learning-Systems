'''
This script implements a dataset for tabular data (diabetes dataset)
'''

from torch.utils.data import Dataset


'''
Class which implements dataset for tabular data
initialize with: 
    x: list --> list of x values
    y: list --> list of labels
'''
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
