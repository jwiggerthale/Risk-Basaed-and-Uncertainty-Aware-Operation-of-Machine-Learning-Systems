'''
this script implements different utils to evaluate models
'''

'''
function to get wba and accuracy from model predictions
'''
def get_wba_acc( 
                labels: list, 
                pred_cls: list, 
                cost_mat: np.array,
                ):
    # Confusion matrix
    cm = get_cm(preds = pred_cls, 
                labels = labels)
    accuracy = acc(preds = pred_cls, 
                   labels=labels)
    er = expected_risk(P = cm, 
                       C = cost_mat)
    max_er = er_max(cost_mat=cost_mat)
    wba = weighted_balanced_accuracy(er = er, 
                                     er_worst=max_er)
    return accuracy, wba



'''
function to get accuracy from predicted classes
'''
def acc(preds: list, 
        labels: list):
    acc = (np.array(preds) == np.array(labels)).sum()/len(labels)
    return acc


'''
function to get maximum er given a cost matrix
'''
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
                          y_pred=preds)
    sums = cm.sum(axis=1, keepdims=True)  # Using keepdims instead of transpose
    # Avoid division by zero - if a class has no samples, keep row as zeros
    # or you could use a small epsilon
    sums = np.where(sums == 0, 1, sums)  # Replace 0 with 1 to avoid division by zero
    cm = cm / sums
    return cm


'''
this function implements calculation of expected risk 
call with: 
  P: np.array --> class conditional probabilities (from confusion matrix)
  C: np.array --> confusion matrix for classification problem
returns: 
  er: float
'''
def expected_risk(P: np.array, 
       C: np.array):
    assert P.shape == C.shape and P.shape[0] == P.shape[1]
    num_classes = P.shape[1]
    er = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            er = er + P[i, j] * C[i, j]
    return er


'''
function to calculate WBA
'''
def weighted_balanced_accuracy(er: float, 
        er_worst: float):
    wba = 1 - er / er_worst
    return wba


'''
function to get optimal decision
call with: 
  preds: np.array --> predicted probabilities
  cost_mat: np.arrray --> cost matrix for classification problem
returns: 
  chosen_labels --> predicted classes that incur least expected prediction risk
  epr --> epr values for chosen classes
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
    


'''
function to calculate abstention statistsics
call with: 
  pred_cls_rn --> predicted classes of rn model
  pred_cls_en --> predicted classes of en model
  eus_rn --> predicted eus of rn model # can also be total predictive uncertainty
  eus_en --> predicted eus of en model # can also be total predictive uncertainty
  epr_en --> predicted epr of en model
  epr_rn --> predicted epr of rn model
   labels --> true labels
  cost_mat --> predicted classes of rn model
  t_eu_en --> threshold for eu of en
  t_eu_rn --> threshold for eu of rn 
  t_epr_en --> threshold for epr of en 
  t_epr_rn --> threshold for epr of rn
returns: 
  stats: dictionary 
    keys: 
       'correct_abstain': 0, #abstained and prediction  had been wrong
        'wrong_abstain': 0, 
        'correct_no_abstain': 0, #model does not abstain and pred is correct
        'wrong_no_abstain': 0
'''
def abstention_stats(pred_cls_rn, 
                    pred_cls_en,
                    eus_rn, 
                    eus_en, 
                    epr_en, 
                    epr_rn,
                     labels,
                    cost_mat, 
                    t_eu_en, 
                    t_eu_rn, 
                    t_epr_en, 
                    t_epr_rn):
    stats = {
        'correct_abstain': 0, #abstained and prediction  had been wrong
        'wrong_abstain': 0, 
        'correct_no_abstain': 0, #model does not abstain and pred is correct
        'wrong_no_abstain': 0
    }
    used_preds = []
    used_labels = []
    for i, p_rn in enumerate(pred_cls_rn):
        p_en = pred_cls_en[i]
        eu_rn = eus_rn[i]
        eu_en = eus_en[i]
        ep_rn = epr_rn[i]
        ep_en = epr_en[i]
        label = labels[i]
        if eu_rn > t_eu_rn:
            abst = True
        elif eu_en > t_eu_en:
            abst = True
        elif ep_rn > t_epr_rn:
            abst = True
        elif ep_en > t_epr_en:
            abst = True
        elif p_en != p_rn:
            abst = True
        else:
            abst = False
            used_preds.append(p_rn)
            used_labels.append(label)
        if abst == False and p_rn == label:
            stats['correct_no_abstain'] += 1
        elif abst == False and p_rn != label:
            stats['wrong_no_abstain'] += 1
        elif abst == True and p_rn == label and p_en == label:
            stats['wrong_abstain'] += 1
        elif abst == True and (p_rn != label or p_en != label):
            stats['correct_abstain'] += 1
    accuracy, wba = get_wba_acc( 
                labels = used_labels,  
                pred_cls = used_preds,
                cost_mat = cost_mat)
    stats['wba'] = wba
    stats['acc'] = accuracy
    return stats
            
        
