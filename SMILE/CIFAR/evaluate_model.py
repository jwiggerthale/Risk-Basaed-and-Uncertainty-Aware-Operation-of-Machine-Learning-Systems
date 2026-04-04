'''
This file implements evaluation of models trained on the CIFAR-100 dataset accoring to the SMILE frmaework
Note that it is based on simple ensemble models as trained for the experiments conducted in Chapters 5 and 6 of the thesis
No implementation of specific training scripts etc. is therefore necessary
You can just evaluate models following the scripts implemented in the folder risk and load the predictions for purpose of evaluation
'''

# imports
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import confusion_matrix
from ..utils import *
from cifar_utils import *


'''
function to get optimal decision under RBD 
call with: 
  preds: np.array --> model predictions
  cost_mat: np.array --> cost matrix for classification problem
'''
def get_optimal_decision(preds: np.array, 
                         cost_mat: np.array):
    chosen_labels = []
    all_epr = []
    for pred in preds:
        best_cost = np.inf
        for i, p in enumerate(pred):
            costs = cost_mat[:, i]
            costs = costs * pred
            costs = costs.sum()
            if costs  < best_cost:
                chosen = i
                best_cost = costs
        chosen_labels.append(chosen)
        all_epr.append(best_cost)
    return np.array(chosen_labels, dtype = float), np.array(all_epr, dtype = float)
    




c = 10
cost_mat = get_cost_matrix_cifar(c)



# exemplary values, adapt to your model
t_eu_en = 0.98
t_epr_en = 5.15
t_eu_rn = 1.73
t_epr_rn = 7.40


# get statistsics for ID test data and CIFAR-100C data (per severity)
all_stats = {}
for s in ['test', 'sev_1', 'sev_2', 'sev_3', 'sev_4', 'sev_5']:
    usecols = ['accuracy', 'er', 'wba']
    # get predictions for en model
    model_stats = {}
    c_dir = f'./data/CIFAR/res_en/{s}'
    preds_test_en = np.load(f'{c_dir}/preds.npy')
    labels_test_en = np.load(f'{c_dir}/labels.npy')
    sigmas_test_en = np.load(f'{c_dir}/sigmas.npy')
    eus_test_en = np.load(f'{c_dir}/eus.npy')
    pred_cls_test_en, all_epr_test_en = get_optimal_decision(preds = preds_test_en, 
                                    cost_mat = cost_mat)
    is_correct_test_en = (labels_test_en == pred_cls_test_en)
    use_sigmas_test_en = np.array([sigmas_test_en[i][int(cls)] for i, cls in enumerate(pred_cls_test_en)])
    
    # get predictions for rn model
    c_dir = f'./data/CIFAR/res_rn/{s}'
    preds_test_rn = np.load(f'{c_dir}/preds.npy')
    labels_test_rn = np.load(f'{c_dir}/labels.npy')
    sigmas_test_rn = np.load(f'{c_dir}/sigmas.npy')
    eus_test_rn = np.load(f'{c_dir}/eus.npy')
    pred_cls_test_rn, all_epr_test_rn = get_optimal_decision(preds = preds_test_rn, 
                                    cost_mat = cost_mat)
    
    is_correct_test_rn = (labels_test_rn == pred_cls_test_rn)
    use_sigmas_test_rn = np.array([sigmas_test_rn[i][int(cls)] for i, cls in enumerate(pred_cls_test_rn)])

    # get stats
    stats = abstention_stats(pred_cls_rn = pred_cls_test_rn, 
                        pred_cls_en = pred_cls_test_en,
                        eus_rn = eus_test_rn, 
                        eus_en = eus_test_en, 
                        epr_en = all_epr_test_en, 
                        epr_rn = all_epr_test_rn,
                         labels = labels_test_en,
                        cost_mat = cost_mat, 
                        t_eu_en = t_eu_en, 
                        t_eu_rn = t_eu_rn, 
                        t_epr_en = t_epr_en, 
                        t_epr_rn = t_epr_rn)
    all_stats[s] = stats



# get omegas and false alarm rates
omegas =  []
fars = []
for key, value in all_stats.items():
    omega = (value['correct_abstain'] + value['wrong_abstain'] )/(value['correct_abstain'] + value['wrong_abstain'] +value['correct_no_abstain'] + value['wrong_no_abstain'])
    omegas.append(omega)
    far = value['wrong_abstain'] /(value['correct_abstain']  + value['wrong_abstain']) 
    fars.append(far)
    all_stats[key]['far'] = far
    all_stats[key]['omega'] = omega


with open('./data/CIFAR/results.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(all_stats, out_file)






