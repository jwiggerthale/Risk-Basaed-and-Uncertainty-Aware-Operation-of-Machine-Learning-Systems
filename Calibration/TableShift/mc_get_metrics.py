'''
This script implements evaluation of MC model
metrics per dataset are calculated and stored
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import copy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from modules.data_utils import heart_ds
from modules.metrics import *
from mc.modules.modules import heart_model
from modules.train_utils import train_loop
import json
import argparse
from torchvision import transforms


parser = argparse.ArgumentParser()


parser.add_argument(
    "--out_dir",
    default="./mc/calibration_results",
    help="Where to store (default: ./mc/calibration_results)",
)

parser.add_argument(
    "--save_dir",
    default="./mc/calibration_models/model_diabetes_survey_seed_2.pth",
    help="Where trained model is stored",
)

args = parser.parse_args()



'''
function to get prediction with MC dropout
call with: 
    model: nn.Module --> MC model
    x: torch.Tensor --> data batch 
returns: 
    pred_cls --> predicted class (tensor of shape [batch size])
    mean_probs --> mean predicted probs per data point (tensor of shape [batch size, num classes])
    eu --> EU (tensor of shape [batch size])
    mean_sigmas --> AU (tensor of shape [batch size])
'''
@torch.no_grad()
def mc_predict(model: nn.Module, 
               x: torch.Tensor, 
               num_samples: int = 20):
    # 1) set model to eval (BN, ... stays frozen)
    was_training = model.training
    model.eval()
    # 2) enable dropout
    def _enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    model.apply(_enable_dropout)
    # 3) forward passes
    probs = []
    sigmas = []
    for _ in range(num_samples):
        mu, sigma = model.forward(x)                      
        probs.append(torch.softmax(mu, dim=-1).detach().cpu())
        sigmas.append(sigma.detach().cpu())
    probs = torch.stack(probs, dim=0)         # [T, B, C]
    sigmas = torch.stack(sigmas, dim = 0)
    mean_sigmas  = sigmas.mean(dim = 0)
    # 4) aggregate and get uncertainty
    mean_probs = probs.mean(dim=0)             # [B, C]
    pred_cls = mean_probs.argmax(dim=1)
    eu = ((probs-mean_probs)**2).sum(dim = (0,2))
    # 5) back to train mode
    if was_training:
        model.train()
    return pred_cls, mean_probs, eu, mean_sigmas 




with open('idx_train.txt', 'r', encoding = 'utf-8') as out_file:
    idx = out_file.readlines()


# get data loader
idx = [int(elem.replace('\n', '')) for elem in idx]
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

train_loader = DataLoader(train_set, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_set, batch_size = 4096)
val_loader = DataLoader(val_set, batch_size = 4096)
ood_loader = DataLoader(ood_data_set, batch_size = 4096)


# load model and weights
model = heart_model(temperature = 1.0, input_dim = 142, num_classes=2).double().to('cuda')
model.load_state_dict(torch.load(args.save_dir))

model.eval()


# get metrics per dataset and save 
preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
uncertainties = []
for x, y in iter(test_loader):
    x = x.cuda()
    pred, probs, un, sigma = mc_predict(model = model, x = x)
    eus.extend(un.cpu().detach().tolist())
    preds.extend(probs.cpu().detach().tolist())
    labels.extend(y.tolist())
    sigmas.extend(sigma.cpu().detach().tolist())
    pred_cls.extend(pred.cpu().detach().tolist())


print(f'accuracy: {(np.array(pred_cls) == np.array(labels)).sum()/len(labels)}')
uncertainties =  np.array(sigmas).mean(axis = 1) + np.array(eus)



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

skce_dict = p_value_skce_ul(probs = preds, 
                            labels = labels)


aurc, _, _= compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=uncertainties)

aurc_eu, _, _ = compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=eus)

aurc_au, _, _ = compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=np.array(sigmas).max(axis = 1))


nll = compute_nll_multiclass(y_true= labels, 
                             probs = preds)


skce = skce_dict['SKCE_ul']
p_value = skce_dict['p_value']
model_stats['ece_5'] = ece_5
model_stats['ece_10'] = ece_10
model_stats['ece_15'] = ece_15
model_stats['ece_20'] = ece_20
model_stats['ece_40'] = ece_40
model_stats['skce'] = skce
model_stats['skce_p'] = p_value
model_stats['aurc'] = aurc
model_stats['aurc_au'] = aurc_au
model_stats['aurc_eu'] = aurc_eu
model_stats['nll'] = nll




if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

with open (f'{args.out_dir}/model_stats_test.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)


plot_aggregated_calibration_curve(probs = preds, 
                                labels = labels, 
                                n_bins=20,  
                                save_path = f'{args.out_dir}/calibration_curve_test.png')



fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)



abstained_results ={'MLP': abstained_results}
#plot_abstained_prediction(results = abstained_results, 
#                          save_path=f'{args.out_dir}/abstained_prediction_curve_test.png')




preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
for x, y in iter(ood_loader):
    x = x.cuda()
    pred, probs, un, sigma = mc_predict(model = model, x = x)
    eus.extend(un.cpu().detach().tolist())
    preds.extend(probs.cpu().detach().tolist())
    labels.extend(y.tolist())
    sigmas.extend(sigma.cpu().detach().tolist())
    pred_cls.extend(pred.cpu().detach().tolist())

uncertainties = np.array(sigmas).mean(axis = 1) + np.array(eus)

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

skce_dict = p_value_skce_ul(probs = preds, 
                            labels = labels)

aurc, _, _ = compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=uncertainties)

aurc_eu, _, _ = compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=eus)

aurc_au, _, _ = compute_aurc(y_true=labels, 
                    y_pred = pred_cls, 
                    uncertainty=np.array(sigmas).max(axis = 1))


nll = compute_nll_multiclass(y_true= labels, 
                             probs = preds)


skce = skce_dict['SKCE_ul']
p_value = skce_dict['p_value']
model_stats['ece_5'] = ece_5
model_stats['ece_10'] = ece_10
model_stats['ece_15'] = ece_15
model_stats['ece_20'] = ece_20
model_stats['ece_40'] = ece_40
model_stats['skce'] = skce
model_stats['skce_p'] = p_value
model_stats['aurc'] = aurc
model_stats['aurc_au'] = aurc_au
model_stats['aurc_eu'] = aurc_eu
model_stats['nll'] = nll

if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

with open (f'{args.out_dir}/model_stats_ood.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)


plot_aggregated_calibration_curve(probs = preds, 
                                labels = labels, 
                                n_bins=20,  
                                save_path = f'{args.out_dir}/calibration_curve_ood.png')


fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)


abstained_results ={'Softmax': abstained_results}
#plot_abstained_prediction(results = abstained_results, 
#                          save_path=f'{args.out_dir}/abstained_prediction_curve_aug.png')



