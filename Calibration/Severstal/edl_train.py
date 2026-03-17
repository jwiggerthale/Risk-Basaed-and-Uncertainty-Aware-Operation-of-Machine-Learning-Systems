'''
This script implements training of EDL model for Severstal dataset
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
from modules.cifar_data_utils import get_cifar_files, class_map, image_dataset, transform_aug, transform_test, transform_train, transform_val
from modules.data_utils import image_dataset, image_dataset_pd, transform_test, transform_train, transform_val, transform_aug
from utils import set_seed

from edl.models import VGG16, resnet18, efficientnet_b0
from edl.edl_basics import evidential_classification
import pandas as pd


    
'''
function to train model
call with:
  model
  optimizer
  scheduler
  train_loader
  val_loader
  num_epochs
  early_stopping --> when to stop model training when no improvement occurs
  out_dir --> where models are stored and log file is written
function trains model for num_epochs and saves model whenever improvement occurs
model is saved to out_dir; additionally out_dir/protocol.txt is created with information on model training 
'''
def train(model: nn.Module, 
          optimizer: nn.Module, 
          scheduler: nn.Module,
          train_loader: DataLoader, 
          val_loader: DataLoader,
          aug_loader: DataLoader = None,
          num_epochs: int = 100,
          early_stopping: int = 10, 
          out_dir: str = 'EDL_Model'  
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
            pred = model.forward(x)
            loss = evidential_classification(pred, y, lamb=min(1.0, epoch / 10))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        if aug_loader is not None: 
            running_loss *= len(train_loader) # ensure valid running loss (divide by 2 x len(train loader in the end))
            for x, y, _ in iter(train_loader):
                x = x.to('cuda')
                y = y.to('cuda')
                pred = model.forward(x)
                loss = evidential_classification(pred, y, lamb=min(1.0, epoch / 10))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            running_loss /= (2 * len(train_loader))
        test_samples = 0
        model.eval()
        for x, y, _ in iter(val_loader):
            x = x.to('cuda')
            y = y.to('cuda')
            with torch.no_grad():
                pred = model.forward(x)
            loss = evidential_classification(pred, y, min(1, epoch / 10))
            test_loss += loss.item()
            pred_cls = pred.argmax(dim = 1)
            acc += (pred_cls == y).sum().item()
            test_samples += len(y)
        acc /= test_samples
        test_loss /= len(val_loader)
        counter += 1
        if scheduler is not None:
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



seed = 1 
set_seed(seed)


# get data loader
train_files = pd.read_csv('train_files_large.csv')
val_files = pd.read_csv('val_files.csv')
test_files = pd.read_csv('test_files.csv')


train_files = train_files.sample(frac=1).reset_index(drop=True)

train_set = image_dataset_pd(image_files=train_files, 
                            transforms=transform_train)
aug_set = image_dataset_pd(image_files=train_files, 
                            transforms=transform_aug)

val_set = image_dataset_pd(image_files=val_files, 
                            transforms=transform_val)


test_set = image_dataset_pd(image_files=test_files, 
                            transforms=transform_test)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
aug_loader = DataLoader(aug_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)




# get model and train
model = resnet18(num_classes=5).to('cuda')
#model = efficientnet_b0(num_classes=5).to('cuda')
#model = VIT(num_classes=100).to('cuda')
#model = VGG16(num_classes=100).to('cuda')
#model.freeze_weights(layers=[1,2,3])

#model = get_repvgg().to('cuda')

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)

train(model=model, 
      optimizer=optimizer, 
      scheduler=scheduler,
      train_loader=train_loader, 
      val_loader=val_loader, 
      num_epochs=300,
      out_dir='./edl/resnet_high_lr_reg_1')
