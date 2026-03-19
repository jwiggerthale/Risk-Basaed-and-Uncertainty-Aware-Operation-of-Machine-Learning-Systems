'''
This script implements evaluation of models with simple softmax output
Results are calculated and stored
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from modules.data_utils import heart_ds
from modules.metrics import *
from mc.modules.modules import heart_model
from modules.train_utils import train_loop
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--out_dir",
    default="./softmax/calibration_fontsizse",
    help="Where to store (default: ./softmax/softmax_calibration_results_balanced)",
)

parser.add_argument(
    "--save_dir",
    default="./ensemble/model_1/model_diabetes_survey_seed_2.pth",
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

train_loader = DataLoader(train_set, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_set, batch_size = 4096)
val_loader = DataLoader(val_set, batch_size = 4096)
ood_loader = DataLoader(ood_data_set, batch_size = 4096)


# get model and load weights
model = heart_model(temperature = 1.0, input_dim = 142, num_classes=2).double().to('cuda')
#model.load_state_dict(torch.load(args.save_dir))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loop(train_loader = train_loader, 
                val_loader = test_loader, 
                model = model, 
                optimizer = optimizer,
                criterion = nn.BCEWithLogitsLoss(),
                device = 'cuda', 
                model_name = f'./softmax/tuned.pth', 
                num_epochs=1,
                early_stopping = 2)




# get predictions, evaluate and store results 
preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
uncertainties = []
for x, y in iter(test_loader):
    x = x.cuda()
    mu, sigma = model.forward(x)
    preds.extend(torch.softmax(mu, dim=-1).detach().cpu().tolist())
    pred = mu.argmax(dim = 1)
    labels.extend(y.tolist())
    pred_cls.extend(pred.cpu().detach().tolist())
    entropy = -(mu.clamp_min(1e-8) * mu.clamp_min(1e-8).log()).sum(dim=1)
    uncertainties.extend(entropy.cpu().detach().numpy())


uncertainties = np.array(uncertainties)
pred_cls = np.array(preds).argmax(axis = 1)
is_wrong = 1 - (np.array(labels) == pred_cls)

scaled_preds = np.array(preds)
scaled_preds = scaled_preds.max(axis = 1)
diff = scaled_preds.max() - scaled_preds.min()
scaled_preds = scaled_preds - scaled_preds.min()
scaled_preds = scaled_preds / diff




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

#aupr = aupr_from_uncertainty(unc = uncertainties, 
#                             is_wrong = is_wrong)

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
#model_stats['aupr'] = 0
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
                                n_bins=20,  
                                save_path = f'{args.out_dir}/calibration_curve_test.png')



fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)



abstained_results ={'ResNet18': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/abstained_prediction_curve_test.png')




preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
for x, y in iter(ood_loader):
    x = x.cuda()
    mu, sigma = model.forward(x)
    preds.extend(torch.softmax(mu, dim=-1).detach().cpu().tolist())
    pred = mu.argmax(dim = 1)
    labels.extend(y.tolist())
    pred_cls.extend(pred.cpu().detach().tolist())
    entropy = -(mu.clamp_min(1e-8) * mu.clamp_min(1e-8).log()).sum(dim=1)
    uncertainties.extend(entropy.cpu().detach().numpy())


uncertainties = np.array(uncertainties)

pred_cls = np.array(preds).argmax(axis = 1)
is_wrong = 1 - (np.array(labels) == pred_cls)


scaled_preds = np.array(preds)
scaled_preds = scaled_preds.max(axis = 1)


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

#aupr = aupr_from_uncertainty(unc = uncertainties, 
#                             is_wrong = is_wrong)


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
#model_stats['aupr'] = aupr
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
                                n_bins=20, 
                                save_path = f'{args.out_dir}/calibration_curve_aug.png')


fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)


abstained_results ={'ResNet18': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/abstained_prediction_curve_aug.png')
