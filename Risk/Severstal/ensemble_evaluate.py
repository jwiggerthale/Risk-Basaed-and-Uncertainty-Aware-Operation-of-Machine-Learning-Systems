'''
This script implements evaluation of CSL ensemble models trained on Severstal dataset
'''



import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from modules.data_utils import image_dataset_pd, transform_test, image_dataset, transform_aug, transform_val
from ensemble.base_models import resnet18, efficientnet_b0
from modules.csl_metrics import get_metrics
from modules.dircihlet_calibration import get_cal
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
    return pred_cls, mean_probs, un, mean_sigmas



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
    "--scale",
    type=int
    default=0,
    help="Where to store (default: ensemble_resnet)",
)

parser.add_argument(
    "--save_dir",
    default="./CSL_2_ensemble_resnet",
    help="Where ensemble models are stored",
)

parser.add_argument(
    "--cost_factor",
    default=2,
    help="Costs for misclassification across super classes",
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
test_files = pd.read_csv('test_files.csv')
val_files = pd.read_csv('val_files.csv')
test_set = image_dataset_pd(image_files=test_files, 
                          transforms=transform_test)
aug_set = image_dataset_pd(image_files = test_files, 
                           transforms=transform_aug)
val_set = image_dataset_pd(image_files=val_files, 
                            transforms=transform_val)

val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

test_loader = DataLoader(test_set, 
                        batch_size=64, 
                        shuffle=False, 
                        drop_last = True)

aug_loader = DataLoader(aug_set, 
                        batch_size=64, 
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
                        batch_size=64, 
                        shuffle=False, 
                        drop_last = True)
        c_sets[f'{method}_{severity}']  = c_loader


# get models and load weights
models = []
for i, f in enumerate(model_files):
    if 'resnet' in args.model:
        model = resnet18(num_classes=5).cuda()
    else:
        model = efficientnet_b0(num_classes=5).cuda()
    model.load_state_dict(torch.load(f))
    model.eval()
    models.append(model)



c= args.cost_factor
if c != -1:
    cost_mat = np.array([
        [0, 1, 1, 1, 1],
        [c, 0, 1, 1, 1],
        [c, 1, 0, 1, 1],
        [c, 1, 1, 0, 1],
        [c, 1, 1, 1, 0]
    ], dtype=float)
else:
    cost_mat = np.array([
        [0, 1, 1/3, 1, 1/2],
        [2, 0, 1/3, 1, 1/2],
        [6, 3, 0, 3, 2/3],
        [2, 1, 1/3, 0, 1/2],
        [4, 2, 2/3, 2, 0]
    ], dtype=float)




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
    
    # get calibrator
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
            cost_mat=cost_mat,
            save_dir=f'{args.save_dir}/temp/test')

preds = []
labels = []
eus = []
sigmas = []
pred_cls = []
for x, y, _ in iter(aug_loader):
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
            cost_mat=cost_mat,
            save_dir=f'{args.save_dir}/temp/aug')



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
    for key, dl in c_sets.items():
        preds = []
        labels = []
        eus = []
        sigmas = []
        pred_cls = []
        for x, y in iter(dl):
            x = x.cuda()
            pred, probs, un, sigma = mc_predict(models = models, x = x)
            eus.extend(un.cpu().detach().tolist())
            all_eus.extend(un.cpu().detach().tolist())
            preds.append(probs.cpu())
            labels.extend(y.tolist())
            sigmas.extend(sigma.cpu().detach().tolist())
            all_sigmas.extend(sigma.cpu().detach().tolist())
            pred_cls.extend(pred.cpu().detach().tolist())
            all_pred_cls.extend(pred.cpu().detach().tolist())
            #all_preds.extend(probs.cpu().detach().tolist())
            all_labels.extend(y.tolist())
            c_eus.extend(un.cpu().detach().tolist())
            c_labels.extend(y.tolist())
            c_pred_cls.extend(pred.cpu().detach().tolist())
            #c_preds.extend(probs.cpu().detach().tolist())
            c_sigmas.extend(sigma.cpu().detach().tolist())
    
        logits = torch.cat(preds, dim=0)
        if args.scale == 1:
            preds = cal.predict_proba(logits) 
        else:
            preds = logits
        all_preds.extend(preds)
        c_preds.extend(preds)

    
    get_metrics(preds=all_preds, 
                labels = all_labels,
                pred_cls = all_pred_cls, 
                sigmas = all_sigmas,
                eus = all_eus, 
                cost_mat = cost_mat,
                save_dir= f'{args.save_dir}/temp/sev_{severity}')


get_metrics(preds=c_preds, 
            labels = c_labels,
            pred_cls = c_pred_cls, 
            sigmas = c_sigmas,
            eus = c_eus, 
            cost_mat=cost_mat,
            save_dir= f'{args.save_dir}/temp/c_combined')
