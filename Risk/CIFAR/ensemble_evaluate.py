'''
This script implements evaluation of CSL ensemble models trained on CIFAR-100 dataset
'''


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from modules.cifar_data_utils import convert_from_c_class, get_cifar_files, class_map, image_dataset, image_dataset_c, transform_test, get_cost_matrix
from ensemble_models.base_models import resnet18, efficientnet_b0
from modules.csl_metrics import get_metrics
from modules.dirichlet_calibration import get_cal
import argparse


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
        mu, sigma = model.forward(x)                      
        probs.append(torch.softmax(mu, dim=-1))
        sigmas.append(sigma)
    probs = torch.stack(probs, dim=0)         # [T, B, C]
    sigmas = torch.stack(sigmas, dim = 0)
    mean_sigmas  = sigmas.mean(dim = 0)
    # 4) aggregate and get uncertainty
    mean_probs = probs.mean(dim=0)             # [B, C]
    pred_cls = mean_probs.argmax(dim=1)
    #un = probs.std(dim = 0).mean(dim = 1)
    # (optional) sinnvollere Unsicherheiten als "std-sum":
    # PrÃ¤diktive Entropie
    #entropy = -(mean_probs.clamp_min(1e-8) * mean_probs.clamp_min(1e-8).log()).sum(dim=1)
    eu = ((probs-mean_probs)**2).sum(dim = (0,2))
    return pred_cls, mean_probs, eu, mean_sigmas




parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="resnet18",
    help="Pick an architecture (default: ResNet18)",
)

parser.add_argument(
    "--out_dir",
    default="./csl_resnet",
    help="Where to store (default: ensemble_resnet)",
)

parser.add_argument(
    "--save_dir",
    default="./CSL_1_effnet",
    help="Where ensemble models are stored",
)

parser.add_argument(
    "--cost_factor",
    default=1,
    help="Costs for misclassification across super classes",
)

parser.add_argument(
    "--scale",
    type=int,
    default=0,
    help="Whether to apply scaling (1 if you want to use scaling)"
)


args = parser.parse_args()

    
# get model files
model_dirs = [f'{args.save_dir}/{f}' for f in os.listdir(args.save_dir) if 'ensemble' in f]
model_files = []
for directory in model_dirs:
    files = [f'{directory}/{f}' for f in os.listdir(directory) if 'model' in f]
    nums = [int(n.split('_')[-1].split('.')[0]) for n in files]
    max_num = max(nums)
    file_name = f'{directory}/model_epoch_{max_num}.pth'
    model_files.append(file_name)
    
    
# get data
train_files, val_files, test_files = get_cifar_files()

test_set = image_dataset(image_files=test_files, 
                        mappings = class_map, 
                        transforms=transform_test)
val_set = image_dataset(image_files=val_files, 
                          mappings = class_map, 
                          transforms=transform_test)

test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    


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


# get models 
models = []
optims = []
schedulers = []
for i, f in enumerate(model_files):
    if 'resnet' in args.model:
        model = resnet18(num_classes=100).cuda()
    else:
        model = efficientnet_b0(num_classes=100).cuda()
    model.load_state_dict(torch.load(f))
    model.eval()
    models.append(model)
    
criterion = nn.CrossEntropyLoss() 
cm = get_cost_matrix(cost_factor=args.cost_factor)

# get logits for val data (calibration)
if args.scale == 1:
    eus = []
    preds = []
    pred_cls = []
    sigmas = []
    labels = []
    for x, y, _ in iter(val_loader):
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


# Evaluation on test data

eus = []
preds = []
pred_cls = []
sigmas = []
labels = []
for x, y, _ in iter(test_loader):
    x = x.cuda()
    pred, probs, un, sigma = mc_predict(models = models, x = x)
    eus.extend(un.cpu().detach().tolist())
    preds.append(probs.cpu())
    labels.extend(y.tolist())
    sigmas.extend(sigma.cpu().detach().tolist())
    pred_cls.extend(pred.cpu().detach().tolist())

logits = torch.cat(preds, dim=0)
if args.scale == 1:
    preds = cal.predict_proba(logits) 
else:
    preds = logits


get_metrics(preds = preds, 
            labels = labels,
            pred_cls = pred_cls, 
            sigmas = sigmas, 
            eus = eus, 
            cost_mat=cm,
            save_dir=f'{args.save_dir}/temp/test')



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
            #print(f'Using key: {key}')
            preds = []
            labels = []
            eus = []
            sigmas = []
            pred_cls = []
            for x, y in iter(dl):
                    x = x.cuda()
                    new_labels = y.tolist()
                    new_labels = [convert_from_c_class[elem] for elem in new_labels]
                    pred, probs, un, sigma = mc_predict(models = models, x = x)
                    eus.extend(un.cpu().detach().tolist())
                    all_eus.extend(un.cpu().detach().tolist())
                    c_eus.extend(un.cpu().detach().tolist())
                    preds.append(probs.cpu())
                    labels.extend(y.tolist())
                    sigmas.extend(sigma.cpu().detach().tolist())
                    all_sigmas.extend(sigma.cpu().detach().tolist())
                    c_sigmas.extend(sigma.cpu().detach().tolist())
                    pred_cls.extend(pred.cpu().detach().tolist())
                    all_pred_cls.extend(pred.cpu().detach().tolist())
                    c_pred_cls.extend(pred.cpu().detach().tolist())
                    #all_preds.extend(probs.cpu().detach().tolist())
                    #c_preds.extend(probs.cpu().detach().tolist())
                    all_labels.extend(new_labels)
                    c_labels.extend(new_labels)
            logits = torch.cat(preds, dim=0)
            if args.scale == 1:
                preds = cal.predict_proba(logits) 
            else:
                preds = logits
            all_preds.extend(preds)
            c_preds.extend(preds)
            uncertainties = np.array(sigmas).mean(axis = 1) + np.array(eus)
            all_uncs.extend(uncertainties)
            c_uncs.extend(uncertainties)
    get_metrics(preds=all_preds, 
                labels = all_labels,
                pred_cls = all_pred_cls, 
                sigmas = all_sigmas,
                eus = all_eus, 
                cost_mat = cm,
                save_dir= f'{args.save_dir}/temp/sev_{severity}')


get_metrics(preds=c_preds, 
            labels = c_labels,
            pred_cls = c_pred_cls, 
            sigmas = c_sigmas,
            eus = c_eus, 
            cost_mat=cm,
            save_dir= f'{args.save_dir}/temp/c_combined')
