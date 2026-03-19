'''
This script implements utils required to train EDL models
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from .edl_basics import evidential_classification
import datetime


def log(message: str, 
        file: str = 'log_file.txt', 
        ):
    now = datetime.datetime.now()
    with open(file, 'a') as out_file:
        out_file.write(f'{now}\t{message}')
        out_file.write('\n')


'''
function to make a train step that automatically utilizes EDL loss
called automatically from train loop function
'''
def train_step(x: torch.tensor,
               y: torch.tensor, 
               device: str,
               model: nn.Module, 
               optimizer: nn.Module, 
               lamb: float = 1.0):
    model.train()
    x, y = x.to(device), y.to(device).long()
    optimizer.zero_grad()
    alpha = model(x)#.flatten()
    loss = evidential_classification(alpha=alpha, y = y, lamb = lamb)
    loss.backward()
    optimizer.step()
    return loss.item()



'''
function to make prediction with model
call with: 
    x: torch.tensor --> image you want to predict on
    model: nn.Module --> model to be applied
returns: 
    preds: torch.tensor --> predicted classes
    logits: torch.tensor --> raw model output
'''
def predict(x: torch.tensor, 
            model: nn.Module):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim = 1)
    return preds, logits



'''
function to make a train step that automatically utilizes EDL loss
called automatically from train loop function
'''
def val_step(x: torch.tensor, 
             y: torch.tensor, 
             model: nn.Module,
             device: str = 'cuda', 
             lamb: float = 1.0):
    x, y = x.to(device), y.to(device).long()
    preds, alpha = predict(model = model, 
                           x = x)
    #preds = preds.flatten()
    loss = evidential_classification(alpha=alpha, y = y, lamb = lamb)
    acc = (preds == y).sum()/len(x)
    wrong = len(x) - (preds == y).sum()
    return loss, acc, wrong



'''
Function to train model in a loop of num epochs
call with: 
    train_loader: DataLoader, --> data loader with train data
    val_loader: DataLoader, --> data loader with validation data
    model: nn.Module, --> model to be trained
    optimizer: nn.Module, --> optimizer to update model weights
    aug_loader: DataLoader = None, --> data loader with augmentations (can be None; was None for the thesis)
    num_epochs: int = 100, --> maximum number of epochs
    early_stopping: int = 20, --> when to interrupt training if no improvement occurs
    log_file: str = 'log.txt', --> where to log results
    device: str = 'cuda', --> devide to use 
    model_name: str = 'heart_model.pth' --> where to store model
'''
def train_loop(train_loader: DataLoader, 
               val_loader: DataLoader, 
               model: nn.Module, 
               optimizer: nn.Module,
               aug_loader: DataLoader = None,
               num_epochs: int = 100, 
               early_stopping: int = 20, 
               log_file: str = 'log.txt', 
               device: str = 'cuda', 
               model_name: str = 'heart_model.pth'):
    best_acc = 0.0
    best_loss = np.inf
    counter = 0
    for epoch in range(num_epochs):
        counter += 1
        running_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        for x, y in iter(train_loader):
            loss = train_step(x = x, 
                              y = y, 
                              model = model, 
                              optimizer = optimizer, 
                              device = device, 
                              lamb=min(0.5, epoch/20))
            running_loss += loss
        running_loss /= len(train_loader)
        if aug_loader is not None:
            running_loss *= len(train_loader)
            for x, y in iter(aug_loader):
                loss = train_step(x = x, 
                                y = y, 
                                model = model, 
                                optimizer = optimizer, 
                                device = device, 
                              lamb=min(0.5, epoch/20))
                running_loss += loss
            running_loss /= (len(train_loader) + len(aug_loader))
        log(f'training in epoch {epoch +1 } completed: ; loss: {running_loss}', 
            file = log_file)
        for x, y in iter(val_loader):
            loss, acc, _ = val_step(x = x, 
                                    y = y, 
                                    model = model, 
                                    device = device, 
                              lamb=min(0.5, epoch/20))
            val_loss += loss
            val_acc += acc
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        log(f'validation in epoch {epoch +1 } completed: acc: {val_acc}; loss: {val_loss}', 
            file=log_file)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f'new best model in epoch {epoch + 1}: accuracy: {val_acc}')
            counter = 0
            torch.save(model.state_dict(), model_name)
        elif val_loss < best_loss:
            best_loss = val_loss
            print(f'new best model in epoch {epoch +1 }: val loss: {val_loss}')
            counter = 0
            torch.save(model.state_dict(), model_name)
        elif(counter > early_stopping):
            print(f'No improvement in {early_stopping} epochs; Training interrupted in epoch {epoch+1}')
            break


   


    
