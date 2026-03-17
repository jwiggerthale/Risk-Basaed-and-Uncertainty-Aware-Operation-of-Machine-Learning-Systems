
'''
This script calculates metrics for EDL model reported in Chapter 5 of the thesis and sotres results to json files
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from modules.cifar_data_utils import get_cifar_files, class_map, image_dataset, image_dataset_c, transform_test, convert_from_c_class
from modules.metrics import *
from edl_modules.models import resnet18, efficientnet_b0
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="resnet18",
    help="Pick an architecture (default: ResNet18)",
)

parser.add_argument(
    "--out_dir",
    default="./edl/resnet_calibration",
    help="Where to store (default: ./edl/resnet_calibration)",
)

parser.add_argument(
    "--save_dir",
    default="./ensemble_resnet",
    help="Where trained model is stored",
)

args = parser.parse_args()


# create datasets
train_files, val_files, test_files = get_cifar_files()


test_set = image_dataset(image_files=test_files, 
                          mappings = class_map, 
                          transforms=transform_test)

test_loader = DataLoader(test_set, 
                        batch_size=64, 
                        shuffle=False, 
                        drop_last = True)
    

# load model 
if 'resnet' in args.model.lower():
    model = resnet18(num_classes=100).to('cuda')
else:
    model = efficientnet_b0(num_classes=100).to('cuda')


# get CIFAR100-C iumages
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



# load model state dict and set to eval
model.load_state_dict(torch.load(args.save_dir))
model.eval()



# get predictions and stats per dataset
preds = []
labels = []
uncertainties = []
pred_cls = []
for x, y, _ in iter(test_loader):
    x = x.cuda()
    pred = model.forward(x)
    sums = pred.sum(axis = 1, keepdim = True)
    probs = pred / sums
    uncs = 100 / pred.sum(axis = 1)
    uncertainties.extend(uncs.cpu().detach().tolist())
    preds.extend(probs.cpu().detach().tolist())
    labels.extend(y.tolist())
    pred_cls.extend(probs.argmax(axis = 1).cpu().detach().tolist())


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


aurc_eu = 0

aurc_au = 0

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



plot_aggregated_calibration_curve(probs = preds, 
                                labels = labels, 
                                n_bins = 20,  
                                save_path = f'{args.out_dir}/{args.model}_calibration_curve_test.png')



fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)


abstained_results ={'EDL': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_test.png')



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
            preds = []
            labels = []
            eus = []
            sigmas = []
            pred_cls = []
            for x, y in iter(dl):
                new_labels = y.tolist()
                new_labels = [convert_from_c_class[elem] for elem in new_labels]
                x = x.cuda()
                pred = model.forward(x)
                sums = pred.sum(axis = 1, keepdim = True)
                probs = pred / sums
                uncs = 100 / pred.sum(axis = 1)
                uncertainties.extend(uncs.cpu().detach().tolist())
                all_uncs.extend(uncs.cpu().detach().tolist())
                c_uncs.extend(uncs.cpu().detach().tolist())
                preds.extend(probs.cpu().detach().tolist())
                all_preds.extend(probs.cpu().detach().tolist())
                c_preds.extend(probs.cpu().detach().tolist())
                labels.extend(new_labels)
                all_labels.extend(new_labels)
                c_labels.extend(new_labels)
                pred_cls.extend(pred.argmax(axis = 1).cpu().detach().tolist())
                all_pred_cls.extend(pred.argmax(axis = 1).cpu().detach().tolist())
                c_pred_cls.extend(pred.argmax(axis = 1).cpu().detach().tolist())



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


    aurc, coverage, risk = compute_aurc(y_true = all_labels, 
                        y_pred = all_pred_cls, 
                        uncertainty=all_uncs)
    nll = compute_nll_multiclass(y_true = np.array(all_labels), 
                                probs = np.array(all_preds))


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


    plot_aggregated_calibration_curve(probs = all_preds, 
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


aurc, coverage, risk = compute_aurc(y_true = c_labels, 
                    y_pred = c_pred_cls, 
                    uncertainty=c_uncs)
nll = compute_nll_multiclass(y_true = np.array(c_labels), 
                            probs = np.array(c_preds))


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



#python3 edl_get_metrics.py --model resnet18 --out_dir ./edl/rn_new --save_dir ./edl/resnet_high_lr_no_reg/model_epoch_8.pth
# python3 edl_get_metrics.py --model effnet --out_dir ./edl/en_new --save_dir ./edl/effnet_high_lr_no_reg/model_epoch_20.pth

if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir, exist_ok=True)

with open (f'{args.out_dir}/model_stats_c_combined_sev_all.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)




plot_aggregated_calibration_curve(probs = c_preds, 
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



