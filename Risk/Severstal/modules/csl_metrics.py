'''
This script implements evaluation of CSL model (based on predictions)
'''

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from .metrics import *
import os
import json


'''
Function whih implements evaluation of model 
Call with: 
    preds: list --> predictions of models
    labels: list, --> true labels
    pred_cls: list, --> predicted classes
    sigmas: list, --> predicted aus
    eus: list, --> predicted eus
    cost_mat: np.array, --> cost matrix for classification problem 
    save_dir: str = './results' --> where to store results

Function    
    - calculates metrics
    - writes them to json file in save_dir
    - plots results to save_dir
    - writes inputs to save_dir
'''
def get_metrics(preds: list, 
                labels: list, 
                pred_cls: list, 
                sigmas: list, 
                eus: list, 
                cost_mat: np.array,
                save_dir: str = './calibration'
                ):
    model_stats = {}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    uncertainties = np.array(sigmas).mean(axis = 1) + np.array(eus)
    # Confusion matrix
    cm = get_cm(preds = pred_cls, 
                labels = labels)
    accuracy = acc(preds = pred_cls, 
                   labels=labels)
    f1 = get_f1_score(preds = preds, labels = labels)
    model_stats['accuracy'] = accuracy
    model_stats['f1'] = f1
    er = expected_risk(P = cm, 
                       C = cost_mat)
    model_stats['er'] = er
    max_er = er_max(cost_mat=cost_mat)
    model_stats['max_er'] = max_er
    wba = weighted_balanced_accuracy(er = er, 
                                     er_worst=max_er)
    model_stats['wba'] = wba
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
    model_stats['ece_5'] = ece_5
    model_stats['ece_10'] = ece_10
    model_stats['ece_15'] = ece_15
    model_stats['ece_20'] = ece_20
    model_stats['ece_40'] = ece_40
    

    skce_dict = p_value_skce_ul(probs = preds, 
                            labels = labels)
    skce = skce_dict['SKCE_ul']
    p_value = skce_dict['p_value']
    model_stats['skce'] = skce
    model_stats['skce_p'] = p_value

    aurc, coverage, risk = compute_aurc(y_true = labels, 
                    y_pred = pred_cls, 
                    uncertainty=uncertainties)
    aurc_eu, coverage_au, risk_au = compute_aurc(y_true = labels, 
                        y_pred = pred_cls, 
                        uncertainty=eus)

    aurc_au, coverage_eu, risk_eu = compute_aurc(y_true = labels, 
                        y_pred = pred_cls, 
                        uncertainty=np.array(sigmas).mean(axis = 1))

    nll = compute_nll_multiclass(y_true = np.array(labels), probs = np.array(preds))
    model_stats['aurc'] = aurc
    model_stats['aurc_eu'] = aurc_eu
    model_stats['aurc_au'] = aurc_au
    model_stats['nll'] = nll
    with open (f'{save_dir}/model_stats_test.json', 'w', encoding = 'utf-8') as out_file:
        json.dump(model_stats, out_file)
    plot_aggregated_calibration_curve(probs = np.array(preds), 
                                    labels = labels, 
                                    n_bins=20,  
                                    save_path = f'{save_dir}/calibration_curve_test.png')
    fractions = np.arange(0.5, 1.01, 0.1)
    abstained_results = abstained_prediction(y_true = labels, 
                                            y_uncertainty=uncertainties, 
                                            probs=preds, 
                                            fractions = fractions)
    abstained_results ={'Ensemble': abstained_results}
    plot_abstained_prediction(results = abstained_results, 
                            save_path=f'{save_dir}/abstained_prediction_curve_test.png')
    preds = np.array(preds)
    np.save(f'{save_dir}/preds.npy', preds)
    labels = np.array(labels)
    np.save(f'{save_dir}/labels.npy', labels)
    eus = np.array(eus)
    np.save(f'{save_dir}/eus.npy', eus)
    sigmas = np.array(sigmas)
    np.save(f'{save_dir}/sigmas.npy', sigmas)
    np.save(f'{save_dir}/cost_mat.npy', cost_mat)
    
    


                             



def er_max(cost_mat: np.array):
    max_cost = 0.0
    for i, elem in enumerate(cost_mat):
        max_cost = max_cost + np.max(elem)
    return max_cost


'''
This function calculates the class conditional probability for each possible outcome and returns it in a numpy array
'''
def get_cm(preds: list,
           labels: list):
    cm = confusion_matrix(y_true=labels, 
                          y_pred = preds)
    sums = np.transpose([cm.sum(axis = 1)])
    cm = cm/sums
    return cm



def expected_risk(P: np.array, 
       C: np.array):
    assert P.shape == C.shape and P.shape[0] == P.shape[1]
    num_classes = P.shape[1]
    er = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            er = er + P[i, j] * C[i, j]
    return er


def weighted_balanced_accuracy(er: float, 
        er_worst: float):
    wba = 1 - er / er_worst
    return wba


def acc(preds: list, 
        labels: list):
    acc = (np.array(preds) == np.array(labels)).sum()/len(labels)
    return acc

    
