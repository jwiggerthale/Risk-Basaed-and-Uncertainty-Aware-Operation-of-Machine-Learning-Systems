

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
from modules.cifar_data_utils import get_cifar_files, class_map, image_dataset, transform_aug, transform_test, transform_train, transform_val
from edl_modules.models import VGG16, resnet18, efficientnet_b0
from edl_modules.edl_basics import evidential_classification


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
            pred = model.forward(x)
            loss = evidential_classification(pred, y, lamb=0)#=min(0.05, epoch / 200))
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
                pred = model.forward(x)
            loss = evidential_classification(pred, y, lamb=0)#min(1, epoch / 10))
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
    


# select your model
model = resnet18(num_classes=100).to('cuda')
#model = efficientnet_b0(num_classes=100).to('cuda')
#model = VIT(num_classes=100).to('cuda')
#model = VGG16(num_classes=100).to('cuda')
#model.freeze_weights(layers=[1,2,3])


criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, min_lr=1e-6)

# train model
train(model=model, 
      optimizer=optimizer, 
      scheduler=None,
      train_loader=train_loader, 
      val_loader=val_loader, 
      num_epochs=300,
      out_dir='./edl/resnet_high_lr_no_reg')
