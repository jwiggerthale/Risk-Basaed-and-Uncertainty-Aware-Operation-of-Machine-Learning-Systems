
'''
This script implements training of MC model

'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
from modules.cifar_data_utils import get_cifar_files, class_map, image_dataset, transform_aug, transform_test, transform_train, transform_val
from modules.base_models import VGG16, VIT, resnet18, efficientnet_b0
from modules.unucertainty_utils import heteroscedastic_ce
from utils import set_seed
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="resnet18",
    help="Pick an architecture (default: ResNet18)",
)

parser.add_argument(
    "--out_dir",
    default="./mc/hetero_resnet",
    help="Where to store (default: ensemble_resnet)",
)

parser.add_argument(
    "--seed",
    type = int,
    default=1,
    help="Seed used for training",
)

args = parser.parse_args()

set_seed(seed = args.seed)

'''
function to train model
call with:  
  model: nn.Module --> model to be trained
  optimizer 
  scheduler
  train_loader
  val_loader
  aug_loader --> implemented to cover augmentation introduced in Chapter 7 of the Thesis; not relevant for benchmarking conducted in Chapter 5
  num_epochs
  early_stopping --> when to interrupt training if no improvement occurs
  out_dir --> directory where models are stored
'''
def train(model: nn.Module, 
          optimizer: nn.Module, 
          scheduler: nn.Module,
          train_loader: DataLoader, 
          val_loader: DataLoader,
          num_epochs: int = 100,
          early_stopping: int = 10, 
          out_dir: str = 'MC_Model'  
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
            loss = heteroscedastic_ce(mu, log_var, y, num_samples=10) #nn.functional.cross_entropy(pred, y)
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
            loss = heteroscedastic_ce(mu, log_var, y, num_samples=10)# nn.functional.cross_entropy(pred, y)
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

        
# get dataloader
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

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    

# get model and train
if 'resnet' in args.model.lower():
    model = resnet18(num_classes=100).to('cuda')
else:
    model = efficientnet_b0(num_classes=100).to('cuda')

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)

train(model=model, 
      optimizer=optimizer, 
      scheduler=scheduler,
      train_loader=train_loader, 
      val_loader=val_loader, 
      out_dir=args.out_dir)
