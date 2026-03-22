
'''
This script examines mean distance between features in Severstal dataset (used in Chapter 4)
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from modules.data_utils import image_dataset_pd, transform_test
from modules.base_models import resnet18, efficientnet_b0
import argparse


parser = argparse.ArgumentParser()

model = 'resnet'
save_dir = "./mc/rn_no_aug.pth"


# get data (important: data loader bathc size needs to be one, adapt fuction to calculate feature distances otherwise)
test_files = pd.read_csv('test_files_large.csv')
test_set = image_dataset_pd(image_files=test_files, 
                          transforms=transform_test)


test_loader = DataLoader(test_set, 
                        batch_size=1, 
                        shuffle=False, 
                        drop_last = True)


# get model and load weights
if 'resnet' in model:
    model = resnet18(num_classes=5).to('cuda')
else:
    model = efficientnet_b0(num_classes=5).to('cuda')
    
model.load_state_dict(torch.load(save_dir))
model.eval()


# get predictions, features and labels
# add features to dict (one list per class)
preds = []
label_cls = []
labels = {i: [] for i in range(5)}
for x, y, _ in iter(test_loader):
    x = x.cuda()
    label_cls.append(y.item())
    mu, _, features = model.forward(x)
    preds.append(mu.argmax().item())
    im_cls = y.item()
    labels[im_cls].append(features.detach().cpu().numpy()[0])


# calculate mean distance and std per class
all_dists = {}
for im_cls in labels: 
    features = np.array(labels[im_cls])
    mean = features.mean(axis = 0)
    dists = features - mean
    dists = dists**2
    dists = dists.sum(axis = 1)
    dist_vals = np.sqrt(dists)
    all_dists[im_cls] = dist_vals



print([val.mean() for val in all_dists.values()])


