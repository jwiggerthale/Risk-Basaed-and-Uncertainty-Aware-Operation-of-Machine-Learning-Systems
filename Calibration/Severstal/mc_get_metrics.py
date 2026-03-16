'''
This script serves to calculate metrics for MC models and store them to json files
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
    default="./ensemble_resnet",
    help="Where to store (default: ensemble_resnet)",
)

parser.add_argument(
    "--save_dir",
    default="./mc/balanced_resnet18.pth",
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
        probs.append(torch.softmax(mu, dim=-1).detach().cpu())
        sigmas.append(sigma.detach().cpu())
    probs = torch.stack(probs, dim=0)         # [T, B, C]
    sigmas = torch.stack(sigmas, dim = 0)
    mean_sigmas  = sigmas.mean(dim = 0)
    # 4) aggregate and get uncertainty
    mean_probs = probs.mean(dim=0)             # [B, C]
    pred_cls = mean_probs.argmax(dim=1)
    eu = ((probs-mean_probs)**2).sum(dim = (0,2))
    # back to training model
    if was_training:
        model.train()
    return pred_cls, mean_probs, eu, mean_sigmas 



# get dataloader
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


# create directory for corrupted data loader
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


# evaluate models and store results
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
model_stats = {}

acc = accuracy(preds = preds, 
               labels = labels)

f1 = get_f1_score(preds=preds, 
                  labels = labels)

model_stats['acc'] = acc
model_stats['f1'] = f1

confs = np.array(preds).max(axis = 1)


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
                                         probs=preds, 
                                         fractions = fractions)



abstained_results ={'MC dropout': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_test.png')









preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
for x, y, _ in iter(aug_loader):
    x = x.cuda()
    pred, probs, un, sigma = mc_predict(model = model, x = x)
    eus.extend(un.cpu().detach().tolist())
    preds.extend(probs.cpu().detach().tolist())
    labels.extend(y.tolist())
    sigmas.extend(sigma.cpu().detach().tolist())
    pred_cls.extend(pred.cpu().detach().tolist())



confs = np.array(preds).max(axis = 1)
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


aurc, _, _ = compute_aurc(y_true = labels, 
                    y_pred = pred_cls, 
                    uncertainty=uncertainties)


aurc_eu, _, _ = compute_aurc(y_true = labels, 
                    y_pred = pred_cls, 
                    uncertainty=eus)

aurc_au, _, _ = compute_aurc(y_true = labels, 
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

with open (f'{args.out_dir}/model_stats_aug.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)


plot_aggregated_calibration_curve(probs = np.array(preds), 
                                labels = labels, 
                                n_bins=20, 
                                save_path = f'{args.out_dir}/{args.model}_calibration_curve_aug.png')

fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = labels, 
                                         y_uncertainty=uncertainties, 
                                         probs=preds, 
                                         fractions = fractions)


abstained_results ={'MC dropout': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                          save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_aug.png')



all_preds = []
all_labels = []
all_eus = []
all_sigmas = []
all_pred_cls = []
for key, dl in c_sets.items():
    preds = []
    labels = []
    eus = []
    sigmas = []
    pred_cls = []
    for x, y in iter(dl):
        x = x.cuda()
        pred, probs, un, sigma = mc_predict(model = model, x = x)
        eus.extend(un.cpu().detach().tolist())
        all_eus.extend(un.cpu().detach().tolist())
        preds.extend(probs.cpu().detach().tolist())
        labels.extend(y.tolist())
        sigmas.extend(sigma.cpu().detach().tolist())
        all_sigmas.extend(sigma.cpu().detach().tolist())
        pred_cls.extend(pred.cpu().detach().tolist())
        all_pred_cls.extend(pred.cpu().detach().tolist())
        all_preds.extend(probs.cpu().detach().tolist())
        all_labels.extend(y.tolist())
    
    

    acc = accuracy(preds = preds, 
                labels = labels)

    f1 = get_f1_score(preds=preds, 
                    labels = labels)

    model_stats['acc'] = acc
    model_stats['f1'] = f1
    confs = np.array(preds).max(axis = 1)
    uncertainties = np.array(sigmas).mean(axis = 1) + np.array(eus)


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

    aurc, _, _= compute_aurc(y_true = labels, 
                        y_pred = pred_cls, 
                        uncertainty=uncertainties)


    aurc_eu, _, _ = compute_aurc(y_true = labels, 
                        y_pred = pred_cls, 
                        uncertainty=eus)

    aurc_au, _, _ = compute_aurc(y_true = labels, 
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

    with open (f'{args.out_dir}/model_stats_{key}.json', 'w', encoding = 'utf-8') as out_file:
        json.dump(model_stats, out_file)


    plot_aggregated_calibration_curve(probs = np.array(preds), 
                                    labels = labels, 
                                    n_bins=20, 
                                    save_path = f'{args.out_dir}/{args.model}_calibration_curve_{key}.png')


    fractions = np.arange(0.5, 1.01, 0.1)
    abstained_results = abstained_prediction(y_true = labels, 
                                            y_uncertainty=uncertainties, 
                                            probs=preds, 
                                            fractions = fractions)


    abstained_results ={'MC dropout': abstained_results}
    plot_abstained_prediction(results = abstained_results, 
                            save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_{key}.png')




uncertainties = np.array(all_sigmas).mean(axis = 1) + np.array(all_eus)
confs = np.array(all_preds).max(axis = 1)

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
                    uncertainty=uncertainties)


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

with open (f'{args.out_dir}/model_stats_c_combined.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(model_stats, out_file)




plot_aggregated_calibration_curve(probs = np.array(all_preds), 
                                labels = all_labels, 
                                n_bins=20, 
                                save_path = f'{args.out_dir}/{args.model}_calibration_curve_c_combined.png')


fractions = np.arange(0.5, 1.01, 0.1)
abstained_results = abstained_prediction(y_true = all_labels, 
                                        y_uncertainty=uncertainties, 
                                        probs=all_preds, 
                                        fractions = fractions)


abstained_results ={'MC dropout': abstained_results}
plot_abstained_prediction(results = abstained_results, 
                        save_path=f'{args.out_dir}/{args.model}_abstained_prediction_curve_c_combined.png')
