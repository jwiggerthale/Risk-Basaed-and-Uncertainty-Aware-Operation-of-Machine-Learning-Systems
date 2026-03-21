from .utils import log
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from .uncertainty_utils import cost_sensitive_heteroscedastic_ce

'''
Function to conduct train step 
called automatically from train loop function
'''
def train_step(x: torch.tensor,
               y: torch.tensor, 
               device: str,
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: nn.Module):
    model.train()
    x, y = x.to(device), y.to(device).double()
    optimizer.zero_grad()
    logits = model(x).flatten()
    loss = criterion(logits, y)
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
            model: nn.Module, 
           inference: bool = False):
    model.eval()
    with torch.no_grad():
        logits = model(x, inference = inference)
        preds = logits.round()
    return preds, logits


'''
Function to conduct val step 
called automatically from train loop function
'''
def val_step(x: torch.tensor, 
             y: torch.tensor, 
             model: nn.Module,
             criterion: nn.Module, 
             device: str = 'cuda'):
    x, y = x.to(device), y.to(device).double()
    preds, probs = predict(model = model, 
                           x = x)
    preds = preds.flatten()
    probs = probs.flatten()
    loss = criterion(probs, y)
    acc = (preds == y).sum()/len(x)
    wrong = len(x) - (preds == y).sum()
    return loss, acc, wrong


'''
Function implements train loop
Model is trained for num_epochs or until early stopping is triggered
Model is stored to model_name
cost sensitive heteroscedastic loss function is applied for training
call with: 
    train_loader: DataLoader, --> data loader with train data
    val_loader: DataLoader, --> data loader with validation data
    model: nn.Module,  --> model to be trained
    optimizer: nn.Module, --> optimizer to adapt model weights
    aug_loader: DataLoader = None, --> data loader with augmented data (not used for Thesis)
    num_epochs: int = 100, --> maximum number of epochs
    early_stopping: int = 20, --> when to interrupt training if no improvementoccurs
    log_file: str = 'log.txt', --> where to write log
    model_name: str = 'heart_model.pth' --> where to store model
'''
def train_loop(train_loader: DataLoader, 
               val_loader: DataLoader, 
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: nn.Module,
               cm: np.array,
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
            optimizer.zero_grad()
            mu, log_var = model.forward(x.to(device).float())
            y = y.to(device)
            loss = cost_sensitive_heteroscedastic_ce(mu=mu, log_var=log_var, target=y.long(),cost_mat=cm, device=device)
            loss.backward()
            optimizer.step()
            running_loss += loss
        running_loss /= len(train_loader)
        if aug_loader is not None:
            running_loss *= len(train_loader)
            for x, y in iter(aug_loader):
                optimizer.zero_grad()
                y = y.to(device)
                mu, log_var = model.forward(x.to(device).float())
                #loss = heteroscedastic_ce(mu = mu, log_var = log_var, target = y)
                loss = cost_sensitive_heteroscedastic_ce(mu=mu, log_var=log_var, target=y.long(),cost_mat=cm, device=device)
                loss.backward()
                optimizer.step()
                running_loss += loss
        log(f'training in epoch {epoch +1 } completed: ; loss: {running_loss}', 
            file = log_file)
        for x, y in iter(val_loader):
            mu, log_var = model.forward(x.to(device).float())
            y = y.to(device)
            loss = cost_sensitive_heteroscedastic_ce(mu=mu, log_var=log_var, target=y.long(),cost_mat=cm, device=device)
            pred_cls = mu.argmax(dim = 1)
            acc = (pred_cls == y).sum()
            val_loss += loss
            val_acc += acc/len(y)
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


   


    
