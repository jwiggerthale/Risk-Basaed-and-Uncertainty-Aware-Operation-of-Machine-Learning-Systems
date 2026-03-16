

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from modules.cifar_data_utils import get_cifar_files, class_map, image_dataset, image_dataset_c, transform_aug, transform_test, convert_from_c_class
from modules.metrics import *
from modules.base_models import resnet18, efficientnet_b0
import argparse
import json


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="effnet",
    help="Pick an architecture (default: ResNet18)",
)

parser.add_argument(
    "--out_dir",
    default="./mc_effnet",
    help="Where to store (default: ensemble_resnet)",
)

parser.add_argument(
    "--save_dir",
    default="./mc/hetero_effnet/model_epoch_40.pth",
    help="Where trained model is stored",
)

args = parser.parse_args()


'''
function to get prediction and uncertainties from ensemble models
call with: 
    model: nn.Module --> ldropout model
    x: torch.Tensor --> data batch 
    num_samples: int --> number of forward passes conducted 
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
        probs.append(torch.softmax(mu, dim=-1))
        sigmas.append(sigma)
    probs = torch.stack(probs, dim=0)         # [T, B, C]
    sigmas = torch.stack(sigmas, dim = 0)
    mean_sigmas  = sigmas.mean(dim = 0)
    # 4) aggregate and get uncertainty
    mean_probs = probs.mean(dim=0)             # [B, C]
    pred_cls = mean_probs.argmax(dim=1)
    eu = ((probs-mean_probs)**2).sum(dim = (0,2))
    # 5) back to train 
    if was_training:
        model.train()
    return pred_cls, mean_probs, eu, mean_sigmas



# get data loader
train_files, val_files, test_files = get_cifar_files()


test_set = image_dataset(image_files=test_files, 
                          mappings = class_map, 
                          transforms=transform_test)

test_loader = DataLoader(test_set, 
                        batch_size=64, 
                        shuffle=False, 
                        drop_last = True)


c_dir = './CIFAR-100-C'
files = [f'{c_dir}/{f}' for f in os.listdir(c_dir) if f.endswith('.npy') and f != 'labels.npy']

label_file = f'{c_dir}/labels.npy'
labels = np.load(label_file)
c_sets = {}
for fp in files:
        ims = np.load(fp)
        for severity in np.arange(1, 6):
            start = (severity-1)*10000
            end = severity * 10000
            c_set = image_dataset_c(ims = ims[start:end], 
                                    labels=labels[start:end], 
                                    transforms=  transform_test)
            c_loader = DataLoader(c_set, 
                            batch_size=16, 
                            shuffle=False, 
                            drop_last = True)
            key = fp.split('/')[-1].split('.')[0]
            key = f'{key}_sev_{severity}'
            c_sets[key]  = c_loader



# get model and load weights
if 'resnet' in args.model.lower():
    model = resnet18(num_classes=100).to('cuda')
else:
    model = efficientnet_b0(num_classes=100).to('cuda')

model.load_state_dict(torch.load(args.save_dir))
model.eval()


# get preds and metrics per dataset
preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
for x, y, _ in iter(test_loader):
    x = x.cuda()
    pred, probs, un, sigma = mc_predict(model = model, x = x)
    eus.extend(un.cpu().detach().tolist())
    preds.extend(probs.cpu().detach().tolist())
    labels.extend(y.tolist())
    sigmas.extend(sigma.cpu().detach().tolist())
    pred_cls.extend(pred.cpu().detach().tolist())

uncertainties = np.array(sigmas).mean(axis = 1) + np.array(eus)
confs = np.array(preds).max(axis = 1)
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
model_stats['aurc_eu'] = aurc_eu
model_stats['aurc_au'] = aurc_au
model_stats['nll'] = nll



if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

with open (f'{args.out_dir}/model_stats_test.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)


plot_aggregated_calibration_curve(probs = np.array(preds), 
                                labels = labels, 
                                n_bins=20,  
                                save_path = f'{args.out_dir}/{args.model}_calibration_curve_test.png')
fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                        y_uncertainty=uncertainties, 
                                        probs = preds, 
                                        fractions = fractions)


abstained_results ={'EDL': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                        save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_c_combined_sev_all.png')


c_preds = []
c_labels = []
c_eus = []
c_sigmas = []
c_pred_cls = []
c_uncs = []
for severity in np.arange(1, 6):
    all_preds = []
    all_labels = []
    all_eus = []
    all_sigmas = []
    all_pred_cls = []
    all_uncs = []
    for key, dl in c_sets.items():
        if f'sev_{severity}' in key:
            print(f'Using key: {key}')
            preds = []
            labels = []
            eus = []
            sigmas = []
            pred_cls = []
            for x, y in iter(dl):
                    x = x.cuda()
                    new_labels = y.tolist()
                    new_labels = [convert_from_c_class[elem] for elem in new_labels]
                    pred, probs, un, sigma = mc_predict(model = model, x = x)
                    eus.extend(un.cpu().detach().tolist())
                    all_eus.extend(un.cpu().detach().tolist())
                    c_eus.extend(un.cpu().detach().tolist())
                    preds.extend(probs.cpu().detach().tolist())
                    labels.extend(new_labels)
                    sigmas.extend(sigma.cpu().detach().tolist())
                    all_sigmas.extend(sigma.cpu().detach().tolist())
                    c_sigmas.extend(sigma.cpu().detach().tolist())
                    pred_cls.extend(pred.cpu().detach().tolist())
                    all_pred_cls.extend(pred.cpu().detach().tolist())
                    c_pred_cls.extend(pred.cpu().detach().tolist())
                    all_preds.extend(probs.cpu().detach().tolist())
                    c_preds.extend(probs.cpu().detach().tolist())
                    all_labels.extend(new_labels)
                    c_labels.extend(new_labels)
            uncertainties = np.array(sigmas).mean(axis = 1) + np.array(eus)
            all_uncs.extend(uncertainties)
            c_uncs.extend(uncertainties)
            confs = np.array(preds).max(axis = 1)
            model_stats = {}
            acc = accuracy(preds = preds, 
                        labels = labels)
            f1 = get_f1_score(preds=preds, 
                            labels = labels)
            model_stats['acc'] = acc
            model_stats['f1'] = f1
            '''

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
            skce = skce_dict['SKCE_ul']
            p_value = skce_dict['p_value']
            model_stats['ece_5'] = ece_5
            model_stats['ece_10'] = ece_10
            model_stats['ece_15'] = ece_15
            model_stats['ece_20'] = ece_20
            model_stats['ece_40'] = ece_40
            model_stats['skce'] = skce
            model_stats['skce_p'] = p_value


            if not os.path.isdir(args.out_dir):
                os.makedirs(args.out_dir, exist_ok=True)

            with open (f'{args.out_dir}/model_stats_{key}.json', 'w', encoding = 'utf-8') as out_file:
                json.dump(model_stats, out_file)


            plot_aggregated_calibration_curve(probs = conf_vals, 
                                            labels = labels, 
                                            n_bins=20, 
                                            save_path = f'{args.out_dir}/{args.model}_calibration_curve_{key}.png')


            fractions = np.arange(0.5, 1.01, 0.1)
            abstained_results = abstained_prediction(y_true = labels, 
                                                    y_uncertainty=uncertainties, 
                                                    probs=preds, 
                                                    fractions = fractions)


            abstained_results ={'EDL': abstained_results}
            plot_abstained_prediction(results = abstained_results, 
                                    save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_{key}.png')
            '''
        else:
            print(f'Skipping key: {key}')
    model_stats = {}
    acc = accuracy(preds = all_preds, 
                labels = all_labels)
    f1 = get_f1_score(preds=all_preds, 
                    labels = all_labels)
    model_stats['acc'] = acc
    model_stats['f1'] = f1


    ece_5 = expected_calibration_error(samples = all_preds, 
                                    true_labels=all_labels, 
                                    M = 5)


    ece_10 = expected_calibration_error(samples = all_preds, 
                                    true_labels=all_labels, 
                                    M = 10)

    ece_15 = expected_calibration_error(samples = all_preds, 
                                    true_labels=all_labels, 
                                    M = 15)

    ece_20 = expected_calibration_error(samples = all_preds, 
                                    true_labels=all_labels, 
                                    M = 20)

    ece_40 = expected_calibration_error(samples = all_preds, 
                                    true_labels=all_labels, 
                                    M = 40)

    skce_dict = p_value_skce_ul(probs = all_preds, 
                                    labels=all_labels)

    aurc, _, _ = compute_aurc(y_true = all_labels, 
                        y_pred = all_pred_cls, 
                        uncertainty=all_uncs)


    aurc_eu, _, _ = compute_aurc(y_true = all_labels, 
                        y_pred = all_pred_cls, 
                        uncertainty=all_eus)

    aurc_au, _, _ = compute_aurc(y_true = all_labels, 
                        y_pred = all_pred_cls, 
                        uncertainty=np.array(all_sigmas).mean(axis = 1))

    nll = compute_nll_multiclass(y_true = np.array(all_labels), probs = np.array(all_preds))


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
    model_stats['aurc_eu'] = aurc_eu
    model_stats['aurc_au'] = aurc_au
    model_stats['nll'] = nll


    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    with open (f'{args.out_dir}/model_stats_c_combined_sev_{severity}.json', 'w', encoding = 'utf-8') as out_file:
        json.dump(model_stats, out_file)




    plot_aggregated_calibration_curve(probs = np.array(all_preds), 
                                    labels = all_labels, 
                                    n_bins=20, 
                                    save_path = f'{args.out_dir}/{args.model}_calibration_curve_c_combined_sev_{severity}.png')


    fractions = np.arange(0.5, 1.01, 0.1)
    abstained_results = abstained_prediction(y_true = all_labels, 
                                            y_uncertainty=all_uncs, 
                                            probs=all_preds, 
                                            fractions = fractions)


    abstained_results ={'EDL': abstained_results}
    plot_abstained_prediction(results = abstained_results, 
                            save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_c_combined_sev_{severity}.png')
    




model_stats = {}

acc = accuracy(preds = c_preds, 
            labels = c_labels)

f1 = get_f1_score(preds=c_preds, 
                labels = c_labels)

model_stats['acc'] = acc
model_stats['f1'] = f1

ece_5 = expected_calibration_error(samples = c_preds, 
                                true_labels=c_labels, 
                                M = 5)


ece_10 = expected_calibration_error(samples = c_preds, 
                                true_labels=c_labels, 
                                M = 10)

ece_15 = expected_calibration_error(samples = c_preds, 
                                true_labels=c_labels, 
                                M = 15)

ece_20 = expected_calibration_error(samples = c_preds, 
                                true_labels=c_labels, 
                                M = 20)

ece_40 = expected_calibration_error(samples = c_preds, 
                                true_labels=c_labels, 
                                M = 40)

skce_dict = p_value_skce_ul(probs = c_preds, 
                                labels=c_labels)


aurc, _, _ = compute_aurc(y_true = c_labels, 
                    y_pred = c_pred_cls, 
                    uncertainty=c_uncs)


aurc_eu, _, _ = compute_aurc(y_true = c_labels, 
                    y_pred = c_pred_cls, 
                    uncertainty=c_eus)

aurc_au, _, _ = compute_aurc(y_true = c_labels, 
                    y_pred = c_pred_cls, 
                    uncertainty=np.array(c_sigmas).max(axis = 1))

nll = compute_nll_multiclass(y_true = np.array(labels), probs = np.array(preds))


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
model_stats['aurc_eu'] = aurc_eu
model_stats['aurc_au'] = aurc_au
model_stats['nll'] = nll


if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

with open (f'{args.out_dir}/model_stats_c_combined_sev_all.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)




plot_aggregated_calibration_curve(probs = np.array(c_preds), 
                                labels = c_labels, 
                                n_bins=20, 
                                save_path = f'{args.out_dir}/{args.model}_calibration_curve_c_combined_sev_all.png')


fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = c_labels, 
                                        y_uncertainty=c_uncs, 
                                        probs = c_preds, 
                                        fractions = fractions)


abstained_results ={'EDL': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                        save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_c_combined_sev_all.png')





