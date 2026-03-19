'''
This script implements training of MC model on diabetes dataset
'''

import pandas as pd
import numpy as np
import json
import copy
import datetime
import matplotlib.pyplot as plt
import random
import os

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from modules.plot_utils import plot_calibration_curve

from mc.modules.modules import heart_model
from modules.data_utils import heart_ds
from modules.train_utils import train_loop, predict
from modules.examination_utils import get_tpr_fpr
from modules.get_scores import get_metrics


# get idx for train data (balanced dataset)
with open('idx_train.txt', 'r', encoding = 'utf-8') as out_file:
    idx = out_file.readlines()

idx = [int(elem.replace('\n', '')) for elem in idx]



# get data loader
x_test = pd.read_csv('./table_shift/Diabetes_X_test.csv')
x_train = pd.read_csv('./table_shift/Diabetes_X_train.csv')
x_train = x_train.iloc[idx, :]
x_train = x_train.reset_index(drop = True)
x_val = pd.read_csv('./table_shift/Diabetes_X_val.csv')
x_ood_test = pd.read_csv('./table_shift/Diabetes_X_ood_test.csv')
x_ood_val = pd.read_csv('./table_shift/Diabetes_X_ood_val.csv')

y_test = pd.read_csv('./table_shift/Diabetes_y_test.csv')
y_train = pd.read_csv('./table_shift/Diabetes_y_train.csv')
y_train = y_train.iloc[idx, :]
y_train = y_train.reset_index(drop = True)
y_val = pd.read_csv('./table_shift/Diabetes_y_val.csv')
y_ood_test = pd.read_csv('./table_shift/Diabetes_y_ood_test.csv')
y_ood_val = pd.read_csv('./table_shift/Diabetes_y_ood_val.csv')

train_set = heart_ds(x = np.array(x_train), y = y_train['DIABETES'].tolist())
val_set = heart_ds(x = np.array(x_val), y = y_val['DIABETES'].tolist())
test_set = heart_ds(x = np.array(x_test), y = y_test['DIABETES'].tolist())
ood_data_set = heart_ds(x = np.array(x_ood_val), y = y_ood_val['DIABETES'].tolist())

train_loader = DataLoader(train_set, batch_size = 128, shuffle=True)
test_loader = DataLoader(test_set, batch_size = 4096)
val_loader = DataLoader(val_set, batch_size = 4096)
ood_loader = DataLoader(ood_data_set, batch_size = 4096)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



# train model for different seed
for seed in [2, 4, 8, 13, 19 , 22, 31,  34, 44, 50, 53, 61, 68, 71, 75, 83, 86, 90, 97, 101]:
    set_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    model = heart_model(temperature = 1.0, input_dim = 142).double().to('cuda')
    model_name = f'./mc/calibration_models/model_diabetes_survey_seed_{seed}.pth'
    if not os.path.isfile(model_name):
        optimizer = torch.optim.Adam(model.parameters())
        train_loop(train_loader = train_loader, 
                   val_loader = test_loader, 
                   model = model, 
                   criterion = criterion, 
                   optimizer = optimizer,
                   device = 'cuda', 
                   model_name = model_name, 
                   num_epochs=100,
                  early_stopping = 2)
