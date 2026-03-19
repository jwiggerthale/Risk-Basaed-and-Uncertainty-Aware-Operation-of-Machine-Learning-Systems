'''
This script implements calculation of metrics for EDL models trained on diabetes dataset
'''
import pandas as pd
import numpy as np
import json
import copy
import random
import os

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from edl.modules.modules import heart_model
from modules.data_utils import heart_ds
from edl.modules.train_utils import train_loop
from modules.metrics import *

import argparse

parser = argparse.ArgumentParser()



parser.add_argument(
    "--out_dir",
    default="./edl/calibration_new_balanced",
    help="Where to store (default: ./mc/calibration_results)",
)

parser.add_argument(
    "--save_dir",
    default="./edl/calibration_models/model_diabetes_survey_seed_2.pth",
    help="Where trained model is stored",
)

args = parser.parse_args()


# get index for train data (balanced dataset)
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

# get model
seed = 2
model = heart_model(temperature = 1.0, input_dim = 136).double().to('cuda')
model.load_state_dict(torch.load(f'./edl/models/model_diabetes_survey_seed_{seed}.pth'))
model.eval()



# evaluate models on different datasets 
preds = []
labels = []
uncertainties = []
for x, y in iter(test_loader):
    x = x.cuda()
    pred = model.forward(x)
    sums = pred.sum(axis = 1, keepdim = True)
    probs = pred / sums
    uncs = 2 / pred.sum(axis = 1)
    uncertainties.extend(uncs.cpu().detach().tolist())
    preds.extend(probs.cpu().detach().tolist())
    labels.extend(y.tolist())



pred_cls = np.array(preds).argmax(axis = 1)
is_wrong = 1 - (np.array(labels) == pred_cls)



model_stats = {}

acc = accuracy(preds = preds, 
               labels = labels)

print(f'accuracy: {acc}')

f1 = get_f1_score(preds=preds, 
                  labels = labels)

model_stats['acc'] = acc
model_stats['f1'] = f1

ece_5 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 5)


ece_10 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 10)

ece_15 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 15)

ece_20 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 20)

ece_40 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 40)

skce_dict = p_value_skce_ul(probs = preds, 
                            labels = labels)

aupr = aupr_from_uncertainty(unc = uncertainties, 
                             is_wrong = is_wrong)

aurc, _, _ = compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=uncertainties)


nll = compute_nll_multiclass(y_true = labels, probs = preds)

skce = skce_dict['SKCE_ul']
p_value = skce_dict['p_value']
model_stats['ece_5'] = ece_5
model_stats['ece_10'] = ece_10
model_stats['ece_15'] = ece_15
model_stats['ece_20'] = ece_20
model_stats['ece_40'] = ece_40
model_stats['skce'] = skce
model_stats['skce_p'] = p_value
model_stats['aupr'] = aupr
model_stats['aurc'] = aurc
model_stats['aurc_au'] = aurc
model_stats['aurc_eu'] = aurc
model_stats['nll'] = nll



if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

with open (f'{args.out_dir}/model_stats_test.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)



plot_aggregated_calibration_curve(probs = preds, 
                                labels = labels, 
                                n_bins = 100,  
                                save_path = f'{args.out_dir}/calibration_curve_test.png')



fractions = np.arange(0.5, 1.0, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)




#args.out_dir = './edl/effnet_high_lr_reg_1'
#args.model = 'effnet'
abstained_results ={'EDL': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/abstained_prediction_curve_test.png')


preds = []
labels = []
uncertainties = []
for x, y in iter(ood_loader):
    x = x.cuda()
    pred = model.forward(x)
    sums = pred.sum(axis = 1, keepdim = True)
    probs = pred / sums
    uncs = 10 / pred.sum(axis = 1)
    uncertainties.extend(uncs.cpu().detach().tolist())
    preds.extend(probs.cpu().detach().tolist())
    labels.extend(y.tolist())


pred_cls = np.array(preds).argmax(axis = 1)
is_wrong = 1 - (np.array(labels) == pred_cls)


model_stats = {}

acc = accuracy(preds = preds, 
               labels = labels)

f1 = get_f1_score(preds=preds, 
                  labels = labels)

model_stats['acc'] = acc
model_stats['f1'] = f1

ece_5 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 5)


ece_10 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 10)

ece_15 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 15)

ece_20 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 20)

ece_40 = expected_calibration_error(samples = preds, 
                                 true_labels=labels, 
                                 M = 40)

aupr = aupr_from_uncertainty(unc = uncertainties, 
                             is_wrong = is_wrong)

aurc, _, _ = compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=uncertainties)


nll = compute_nll_multiclass(y_true = labels, probs = preds)

skce = skce_dict['SKCE_ul']
p_value = skce_dict['p_value']
model_stats['ece_5'] = ece_5
model_stats['ece_10'] = ece_10
model_stats['ece_15'] = ece_15
model_stats['ece_20'] = ece_20
model_stats['ece_40'] = ece_40
model_stats['skce'] = skce
model_stats['skce_p'] = p_value
model_stats['aupr'] = aupr
model_stats['aurc'] = aurc
model_stats['aurc_au'] = aurc
model_stats['aurc_eu'] = aurc
model_stats['nll'] = nll





if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

with open (f'{args.out_dir}/model_stats_aug.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)



plot_aggregated_calibration_curve(probs = preds, 
                                labels = labels, 
                                n_bins = 100,  
                                save_path = f'{args.out_dir}/calibration_curve_ds.png')


fractions = np.arange(0.5, 1.0, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)


abstained_results ={'EDL': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/abstained_prediction_curve_ds.png')
