'''
This script implements training of CSL (ensemble) models for CIFAR-100 dataset
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from modules.cifar_data_utils import get_cifar_files, class_map, image_dataset, transform_test, transform_train, transform_val, get_cost_matrix
from ensemble_models.base_models import  resnet18, efficientnet_b0
import argparse
from modules.unucertainty_utils import new_cost_sensitive_heteroscedastic_ce
from modules.utils import set_seed


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
    "--cost_factor",
    default=2,
    help="Cost factor for cross super class confusion",
)

parser.add_argument(
    "--seed",
    default=42,
    help="Seed for training",
)

parser.add_argument(
    "--num_models",
    default=5,
    help="Cost factor for cross super class confusion",
)



'''
Function to train model 
call with: 
  model: nn.Module, --> model to be trained 
  optimizer: nn.Module, --> optimizer 
  scheduler: nn.Module, --> learning rate scheduler
  train_loader: DataLoader, --> data loader with train data
  val_loader: DataLoader, --> data loader with validation data
  cm: np.array, --> cost matrix
  num_epochs: int = 100, --> max number of epochs
  early_stopping: int = 10,  --> when to interrupt training if no improvement occurs
  out_dir: str = 'Model' --> where to store results
'''
def train(model: nn.Module, 
          optimizer: nn.Module, 
          scheduler: nn.Module,
          train_loader: DataLoader, 
          val_loader: DataLoader,
          cm: np.array,
          num_epochs: int = 100,
          early_stopping: int = 10, 
          out_dir: str = 'Model'  
          ):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    best_acc = 0.0
    best_loss = np.inf
    counter = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        test_loss = 0.0
        acc = 0.0
        model.train()
        for x, y, _ in iter(train_loader):
            x = x.to('cuda')
            y = y.to('cuda')
            mu, log_var = model.forward(x)
            loss = new_cost_sensitive_heteroscedastic_ce(mu, log_var, y, cost_mat=cm, num_samples=10, alpha = 0.5) # nn.functional.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        test_samples = 0
        model.eval()
        for x, y, _ in iter(val_loader):
            x = x.to('cuda')
            y = y.to('cuda')
            with torch.no_grad():
                mu, log_var = model.forward(x)
            loss = new_cost_sensitive_heteroscedastic_ce(mu, log_var, y, cost_mat=cm, num_samples=10, alpha = 0.5) # nn.functional.cross_entropy(pred, y)
            test_loss += loss.item()
            pred_cls = mu.argmax(dim = 1)
            acc += (pred_cls == y).sum().item()
            test_samples += len(y)
        acc /= test_samples
        test_loss /= len(val_loader)
        counter += 1
        scheduler.step(test_loss)
        print(f'Training in epoch {epoch +1} finished: Acc: {acc}; Test Loss: {test_loss}, Train Loss: {running_loss}')
        with open(f'{out_dir}/protocol.csv', 'a', encoding='utf-8') as out_file:
            out_file.write(f'{epoch+1},{running_loss},{test_loss},{acc}\n')
        if test_loss < best_loss:
            torch.save(model.state_dict(), f'{out_dir}/model_epoch_{epoch+1}.pth')
            best_loss = test_loss
            counter = 0
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{out_dir}/model_epoch_{epoch+1}.pth')
            counter = 0
        elif counter > early_stopping:
            print(f'Early stopping in epoch {epoch +1}')
            break
        

# get data
train_files, val_files, test_files = get_cifar_files()

train_set = image_dataset(image_files=train_files, 
                          mappings = class_map, 
                          transforms=transform_train)

val_set = image_dataset(image_files=val_files, 
                          mappings = class_map, 
                          transforms=transform_val)

test_set = image_dataset(image_files=test_files, 
                          mappings = class_map, 
                          transforms=transform_test)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    

args = parser.parse_args()
set_seed(args.seed)


# create models ad optiizers 
models = []
optims = []
schedulers = []
for _ in range(args.num_models):
    if 'resnet' in args.model:
        model = resnet18(num_classes=100).cuda()
    else:
        model = efficientnet_b0(num_classes=100).cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)
    optims.append(optim)
    models.append(model)
    schedulers.append(scheduler)
    


criterion = nn.CrossEntropyLoss() 
cm = get_cost_matrix(cost_factor=args.cost_factor)
cm = torch.tensor(cm)

# train models
for i in range(args.num_models):
    model = models[i]
    optim = optims[i]
    scheduler = schedulers[i]
    save_dir = f'{args.out_dir}/ensemble_{i}'
    if not os.path.isdir(save_dir):
        model.cuda()
        train(model=model, 
            optimizer=optim, 
            scheduler=scheduler,
            train_loader=train_loader, 
            val_loader=val_loader, 
            num_epochs=100,
            out_dir=save_dir, 
            cm = cm)
        model.cpu()
        torch.cuda.empty_cache()
