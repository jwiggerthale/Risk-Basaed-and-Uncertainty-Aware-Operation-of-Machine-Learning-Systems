'''
This script implements different function to evaluate models
'''



'''
function to calculate expected risk
call with: 
    c_fn: float, --> risk ratio
    fpr_val: float, --> fpr
    fnr_val: float --> fnr
'''
def er(c_fn: float, 
      fpr_val: float, 
      fnr_val: float):
    er = fpr_val + c_fn * fnr_val
    return er



'''
Function to calculate wba (binary)
call with: 
    c_fn: float, --> risk ratio
    fpr_val: float, --> fpr
    fnr_val: float --> fnr
'''
def wba(c_fn: float, 
      fpr_val: float, 
      fnr_val: float):
    wba = ((1 - fpr_val) + c_fn * (1 - fnr_val))/ (1 + c_fn)
    return wba



'''
functions to calculate precision respectively recall
call with: 
    fpr_val: float, --> fpr
    fnr_val: float --> fnr
'''
def precision(tpr_val: float, 
             fpr_val: float):
    precision = tpr_val / (tpr_val + fpr_val)
    return precision

def recall(tpr_val: float, 
          fnr_val: float):
    recall  = tpr_val / (tpr_val + fnr_val)
    return recall


'''
function to calculate f1 score
call with: 
    precision_val: float, --> result obtained fromn precision function
    recall_val: float --> result obtained from recall function
'''
def f1_score(precision_val: float, 
             recall_val: float):
    f1 = 2 * precision_val * recall_val / (precision_val + recall_val)
    return f1



'''
function to calculate different metrics 
call with: 
    c_fn: float, --> risk ratio
    fpr_val: float, --> fpr
    fnr_val: float --> fnr
returns metrics (float each)
    fpr_val, 
    tpr_val, 
    fnr_val, 
    tnr_val, 
    er_val, 
    wba_val, 
    precision_val, 
    recall_val, 
    f1_val
'''
def get_metrics(c_fn: float, 
                fpr_val: float, 
                fnr_val: float):
    tnr_val = 1 - fpr_val
    tpr_val = 1 - fnr_val
    er_val = er(c_fn = c_fn, 
                fpr_val = fpr_val, 
                fnr_val = fnr_val)
    wba_val = wba(c_fn = c_fn, 
                fpr_val = fpr_val, 
                fnr_val = fnr_val)
    precision_val = precision(tpr_val = tpr_val, 
                              fpr_val = fpr_val)
    recall_val = recall(tpr_val = tpr_val, 
                    fnr_val = fnr_val)
    f1_val = f1_score(precision_val = precision_val, 
                      recall_val = recall_val)
    return fpr_val, tpr_val, fnr_val, tnr_val, er_val, wba_val, precision_val, recall_val, f1_val
    
