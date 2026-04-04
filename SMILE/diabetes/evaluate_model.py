import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.metrics import confusion_matrix



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
    


# define base values
base_dir = './data/Severstal'

model_stats = {}

c = 8
cost_mat = np.array([[0, c],[1, 0]]) 
cost_mat_scaled = np.array([[0, c],[6.97, 0]]) 


t_eu_en = 0.89
t_epr_en = 1.0
t_eu_rn = 0.91
t_epr_rn = 0.98

# get stats
all_stats = {}
for s in ['test', 'ds']:
    c_dir = f'./data/diabtes/res_nn/{s}'
    preds_test_en = np.load(f'{c_dir}/preds.npy')
    labels_test_en = np.load(f'{c_dir}/labels.npy')
    sigmas_test_en = np.load(f'{c_dir}/sigmas.npy')
    eus_test_en = np.load(f'{c_dir}/eus.npy')
    pred_cls_test_en, all_epr_test_en = get_optimal_decision(preds = preds_test_en, 
                                    cost_mat = cost_mat)
    
    is_correct_test_en = (labels_test_en == pred_cls_test_en)
    use_sigmas_test_en = np.array([sigmas_test_en[i][int(cls)] for i, cls in enumerate(pred_cls_test_en)])
    
    
    model_type = 'rn'
    c_dir = f'./data/diabtes/res_rf/{s}'
    preds_test_rn = np.load(f'{c_dir}/preds.npy')
    labels_test_rn = np.load(f'{c_dir}/labels.npy')
    sigmas_test_rn = np.load(f'{c_dir}/sigmas.npy')
    eus_test_rn = np.load(f'{c_dir}/eus.npy')
    pred_cls_test_rn, all_epr_test_rn = get_optimal_decision(preds = preds_test_rn, 
                                    cost_mat = cost_mat)

    
    is_correct_test_rn = (labels_test_rn == pred_cls_test_rn)
    use_sigmas_test_rn = np.array([sigmas_test_rn[i][int(cls)] for i, cls in enumerate(pred_cls_test_rn)])

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


with open('./data/diabetes/results.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(all_stats, out_file)
