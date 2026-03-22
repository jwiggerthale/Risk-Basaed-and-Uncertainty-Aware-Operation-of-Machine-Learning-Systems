'''
This script implemnets evaluation of model with dirichlet calibration
'''



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from modules.modules import heart_model
from modules.data_utils import heart_ds
from modules.csl_metrics import get_metrics
from modules.dirichlet_calibration import get_cal
import argparse
import pandas as pd


'''
Function to make prediction with ensemble of models
(called MC_predict since I first implemented scripts for MC dropout and just slightly adapted the logic of the function)
call with: 
    models: list,  --> list of models
    x: torch.Tensor --> input
returns: 
    pred_cls --> predicted class for input
    mean_probs --> mean predicted values 
    eu --> EU
    mean_sigmas --> mean predicted AU
'''
@torch.no_grad()
def mc_predict(models: list, 
               x: torch.Tensor):
    probs = []
    sigmas = []
    for model in models:
        model.eval()
        mu, sigma = model.forward(x.float())                      
        probs.append(torch.softmax(mu, dim=-1))
        sigmas.append(sigma)
    probs = torch.stack(probs, dim=0)         # [T, B, C]
    sigmas = torch.stack(sigmas, dim = 0)
    mean_sigmas  = sigmas.mean(dim = 0)
    mean_probs = probs.mean(dim=0)             # [B, C]
    pred_cls = mean_probs.argmax(dim=1)
    entropy = -(mean_probs.clamp_min(1e-8) * mean_probs.clamp_min(1e-8).log()).sum(dim=1)
    eu = ((probs-mean_probs)**2).sum(dim = (0,2))
    return pred_cls, mean_probs, eu, mean_sigmas




parser = argparse.ArgumentParser()


parser.add_argument(
    "--out_dir",
    default="./ensemble",
    help="Where to store (default: ensemble_resnet)",
)

parser.add_argument(
    "--save_dir",
    default="./ensemble",
    help="Where ensemble models are stored",
)

parser.add_argument(
    "--cost_factor",
    default=2,
    help="Costs for misclassification across super classes",
)

parser.add_argument(
    "--temperature",
    default=2,
    help="Costs for misclassification across super classes",
)

args = parser.parse_args()
t = float(args.temperature)
   
       

# get data
with open('/data/PhDThesis/Calibration/TableShift/idx_train.txt', 'r', encoding = 'utf-8') as out_file:
    idx = out_file.readlines()

idx = [int(elem.replace('\n', '')) for elem in idx]

x_test = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_X_test.csv')
x_train = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_X_train.csv')
x_train = x_train.iloc[idx, :]
x_train = x_train.reset_index(drop = True)
x_val = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_X_val.csv')
x_ood_test = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_X_ood_test.csv')
x_ood_val = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_X_ood_val.csv')

y_test = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_y_test.csv')
y_train = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_y_train.csv')
y_train = y_train.iloc[idx, :]
y_train = y_train.reset_index(drop = True)
y_val = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_y_val.csv')
y_ood_test = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_y_ood_test.csv')
y_ood_val = pd.read_csv('/data/PhDThesis/Calibration/TableShift/table_shift/Diabetes_y_ood_val.csv')

train_set = heart_ds(x = np.array(x_train), y = y_train['DIABETES'].tolist())
val_set = heart_ds(x = np.array(x_val), y = y_val['DIABETES'].tolist())
test_set = heart_ds(x = np.array(x_test), y = y_test['DIABETES'].tolist())
ood_data_set = heart_ds(x = np.array(x_ood_val), y = y_ood_val['DIABETES'].tolist())

train_loader = DataLoader(train_set, batch_size = 128, shuffle=True)
test_loader = DataLoader(test_set, batch_size = 4096)
val_loader = DataLoader(val_set, batch_size = 4096)
ood_loader = DataLoader(ood_data_set, batch_size = 4096)

# Evaluation on test data
for c_fn_val in [1, 2,4,8,16,32]:
    cost_mat = np.array([[0, 1], [c_fn_val, 0]], dtype = float)
    save_dir = f'{args.save_dir}/ensembles_csl_{c_fn_val}'
    model_files = [f'{save_dir}/{f}/model.pth' for f in os.listdir(save_dir) if 'model_' in f]
    models = []
    torch.cuda.empty_cache()
    for i, f in enumerate(model_files):
        model = heart_model(temperature = t, input_dim = 142).to('cuda')
        c_fn = torch.ones([128])*c_fn_val
        c_fn = torch.tensor([c_fn_val], dtype=torch.double)
        cost_mat = np.array([[0, 1], [c_fn_val, 0]])
        model.load_state_dict(torch.load(f))
        model.eval()
        models.append(model)
    eus = []
    preds = []
    pred_cls = []
    sigmas = []
    labels = []
    for x, y in iter(val_loader):
        x = x.cuda()
        pred, probs, un, sigma = mc_predict(models = models, x = x)
        eus.extend(un.cpu().detach().tolist())
        preds.append(probs.cpu())
        labels.append(y)
        sigmas.extend(sigma.cpu().detach().tolist())
        pred_cls.extend(pred.cpu().detach().tolist())

    logits_val = torch.cat(preds, dim=0)
    y_val = torch.cat(labels, dim=0)

    cal = get_cal(y_cal = y_val, 
                logits_cal = logits_val)

    eus = []
    preds = []
    pred_cls = []
    sigmas = []
    labels = []
    for x, y in iter(test_loader):
        x = x.cuda()
        pred, probs, un, sigma = mc_predict(models = models, x = x)
        eus.extend(un.cpu().detach().tolist())
        preds.append(probs.cpu())
        labels.extend(y.tolist())
        sigmas.extend(sigma.cpu().detach().tolist())
        pred_cls.extend(pred.cpu().detach().tolist())
    
    logits = torch.cat(preds, dim=0)
    preds = cal.predict_proba(logits) 

    get_metrics(preds = preds, 
                labels = labels,
                pred_cls = pred_cls, 
                sigmas = sigmas, 
                eus = eus, 
                cost_mat=cost_mat,
                save_dir=f'{save_dir}/results/temp/test')

    preds = []
    labels = []
    eus = []
    sigmas = []
    pred_cls = []
    for x, y in iter(ood_loader):
        x = x.cuda()
        pred, probs, un, sigma = mc_predict(models = models, x = x)
        eus.extend(un.cpu().detach().tolist())
        preds.append(probs.cpu())
        labels.extend(y.tolist())
        sigmas.extend(sigma.cpu().detach().tolist())
        pred_cls.extend(pred.cpu().detach().tolist())

    logits = torch.cat(preds, dim=0)
    preds = cal.predict_proba(logits) 
    get_metrics(preds = preds, 
                labels = labels,
                pred_cls = pred_cls, 
                sigmas = sigmas, 
                eus = eus, 
                cost_mat=cost_mat,
                save_dir=f'{save_dir}/results/temp/aug')



