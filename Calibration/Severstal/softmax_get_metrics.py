'''
This script implements calculation of metris for Severstal dataset with Softmax model
You can use ensemble train function with m=1 to train softmax models 
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from modules.data_utils import image_dataset, image_dataset_pd, transform_aug, transform_test
from modules.metrics import *
from modules.base_models import resnet18, efficientnet_b0
import json
import argparse
from torchvision import transforms


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="resnet18",
    help="Pick an architecture (default: ResNet18)",
)

parser.add_argument(
    "--out_dir",
    default="./softmax/resnet_calibration",
    help="Where to store (default: ensemble_resnet)",
)

parser.add_argument(
    "--save_dir",
    default="./mc/rn_no_aug.pth",
    help="Where trained model is stored",
)

args = parser.parse_args()


# get data loader
test_files = pd.read_csv('test_files_large.csv')
test_set = image_dataset_pd(image_files=test_files, 
                          transforms=transform_test)

aug_set = image_dataset_pd(image_files = test_files, 
                           transforms=transform_aug)

test_loader = DataLoader(test_set, 
                        batch_size=16, 
                        shuffle=False, 
                        drop_last = True)

aug_loader = DataLoader(aug_set, 
                        batch_size=16, 
                        shuffle=False, 
                        drop_last = True)


d = {}
d['Gaussian Noise'] = 'gaussian_noise'
d['Shot Noise'] = 'shot_noise'
d['Impulse Noise'] = 'impulse_noise'
d['Speckle Noise'] = 'speckle_noise'
d['Gaussian Blur'] = 'gaussian_blur'
d['Defocus Blur'] = 'defocus_blur'
#d['Motion Blur'] = 'motion_blur'
d['Zoom Blur'] = 'zoom_blur'
d['Contrast'] = 'contrast'
d['Pixelate'] = 'pixelate'
d['JPEG'] = 'jpeg_compression'
d['Elastic'] = 'elastic_transform'

c_files = {}
c_sets = {}
for name, method in d.items():
    for severity in range(1,6):
        fp = f'severstal_corrupted/{method}/images_severity_{severity}.npy'
        ims = np.load(fp)
        fp = f'severstal_corrupted/{method}/labels_severity_{severity}.npy'
        labels = np.load(fp)
        c_files[f'{method}_{severity}'] = [ims, labels]
        c_set = image_dataset(ims = ims, 
                                labels=labels, 
                                transforms=  transforms.Compose([transforms.ToTensor()]))
        c_loader = DataLoader(c_set, 
                        batch_size=16, 
                        shuffle=False, 
                        drop_last = True)
        c_sets[f'{method}_{severity}']  = c_loader



# get model and load weights
if 'resnet' in args.model.lower():
    model = resnet18(num_classes=5).to('cuda')
else:
    model = efficientnet_b0(num_classes=5).to('cuda')
    model.set_dropout()



model.load_state_dict(torch.load(args.save_dir))
model.eval()




# get predictions per dataset, calculate metrics and store results
preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
uncertainties = []
for x, y, _ in iter(test_loader):
    x = x.cuda()
    mu, sigma = model.forward(x)
    preds.extend(torch.softmax(mu, dim=-1).detach().cpu().tolist())
    pred = mu.argmax(dim = 1)
    labels.extend(y.tolist())
    pred_cls.extend(pred.cpu().detach().tolist())
    entropy = -(mu.clamp_min(1e-8) * mu.clamp_min(1e-8).log()).sum(dim=1)
    uncertainties.extend(entropy.cpu().detach().numpy())



uncertainties = np.array(uncertainties)


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
                                n_bins=20,  
                                save_path = f'{args.out_dir}/{args.model}_calibration_curve_test.png')



fractions = np.arange(0.5, 1.0, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)



abstained_results ={'ResNet18': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_test.png')




preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
for x, y, _ in iter(aug_loader):
    x = x.cuda()
    mu, sigma = model.forward(x)
    preds.extend(torch.softmax(mu, dim=-1).detach().cpu().tolist())
    pred = mu.argmax(dim = 1)
    labels.extend(y.tolist())
    pred_cls.extend(pred.cpu().detach().tolist())
    entropy = -(mu.clamp_min(1e-8) * mu.clamp_min(1e-8).log()).sum(dim=1)
    uncertainties.extend(entropy.cpu().detach().numpy())



uncertainties = np.array(uncertainties)

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

with open (f'{args.out_dir}/model_stats_aug.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)


plot_aggregated_calibration_curve(probs = preds, 
                                labels = labels, 
                                n_bins=20, 
                                save_path = f'{args.out_dir}/{args.model}_aug.png')


fractions = np.arange(0.5, 1.0, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)


abstained_results ={'ResNet18': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_aug.png')




all_preds = []
all_labels = []
all_eus = []
all_sigmas = []
all_pred_cls = []
all_uncertainties = []
for key, dl in c_sets.items():
    preds = []
    labels = []
    eus = []
    sigmas = []
    pred_cls = []
    for x, y in iter(dl):
        x = x.cuda()
        mu, sigma = model.forward(x)
        preds.extend(torch.softmax(mu, dim=-1).detach().cpu().tolist())
        all_preds.extend(torch.softmax(mu, dim=-1).detach().cpu().tolist())
        pred = mu.argmax(dim = 1)
        labels.extend(y.tolist())
        all_labels.extend(y.tolist())
        pred_cls.extend(pred.cpu().detach().tolist())
        all_pred_cls.extend(pred.cpu().detach().tolist())
        entropy = -(mu.clamp_min(1e-8) * mu.clamp_min(1e-8).log()).sum(dim=1)
        uncertainties.extend(entropy.cpu().detach().numpy())



    uncertainties = np.array(uncertainties)
    all_uncertainties.extend(uncertainties)
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

    with open (f'{args.out_dir}/model_stats_{key}.json', 'w', encoding = 'utf-8') as out_file:
        json.dump(model_stats, out_file)


    plot_aggregated_calibration_curve(probs = preds, 
                                    labels = labels, 
                                    n_bins=20, 
                                    save_path = f'{args.out_dir}/{args.model}_calibration_curve_{key}.png')


    fractions = np.arange(0.5, 1.0, 0.1)
    abstained_results = abstained_prediction(y_true = labels, 
                                            y_uncertainty=uncertainties, 
                                            probs=preds, 
                                            fractions = fractions)


    abstained_results ={'EDL': abstained_results}
    plot_abstained_prediction(results = abstained_results, 
                            save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_{key}.png')




uncertainties = np.array(all_uncertainties)

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
                    uncertainty=uncertainties)
aurc_eu = 0
aurc_au = 0
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

with open (f'{args.out_dir}/model_stats_c_combined.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)



plot_aggregated_calibration_curve(probs = all_preds, 
                                labels = all_labels, 
                                n_bins=20, 
                                save_path = f'{args.out_dir}/{args.model}_calibration_curve_c_combined.png')


fractions = np.arange(0.5, 1.0, 0.1)
abstained_results = abstained_prediction(y_true = all_labels, 
                                        y_uncertainty=uncertainties, 
                                        probs=all_preds, 
                                        fractions = fractions)


abstained_results ={'EDL': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                        save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_c_combined.png')
