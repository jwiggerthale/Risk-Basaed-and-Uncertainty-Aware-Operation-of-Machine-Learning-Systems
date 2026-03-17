'''
This file implements different EDL models for classification in PyTorch
Models are based on PyTorch implementations and use Dirichlet layer as final classification layer 
Models are initialized with ImageNet weights (stored locally)
'''


import torch
import torch.nn as nn
from torchvision import models
from .edl_basics import Dirichlet




class VGG16(nn.Module):
    def __init__(self, 
                num_classes: int = 6):
        super().__init__()
        model = models.vgg16()
        model.load_state_dict(torch.load('/data/Models/image_recognition/vgg16_pretrained.pth'))
        self.feature_extractor = model.features
        self.pooler = model.avgpool
        self.clf = model.classifier
        self.clf[6] = Dirichlet(4096, num_classes)
    # freeze feature extractor
    def freeze_weights(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
    def forward(self, 
                x: torch.tensor):
        pred = self.feature_extractor(x)
        pred = self.pooler(pred)
        pred = pred.reshape(-1, 25088)
        pred = self.clf(pred)
        return pred



class efficientnet_b0(nn.Module):
    def __init__(self, 
                num_classes: int = 5):
        super().__init__()
        model = models.efficientnet_b0(pretrained = False)
        model.load_state_dict(torch.load('/data/Models/image_recognition/efficientnet_b0.pth'))
        self.model = model
        self.model.classifier =  Dirichlet(1280, num_classes)
    # freeze feature extractor
    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    # freeze certain layers
    def freeze_layers(self, 
                      num_frozen_blocks: int = 9):
        for i in range(num_frozen_blocks):
            for param in self.model.encoder.layers[i].parameters():
                param.requires_grad = False
    def set_dropout(self, 
                    p=0.2):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = p
    def forward(self, 
                x: torch.tensor):
        pred = self.model(x)
        return pred



class resnet18(nn.Module):
    def __init__(self, 
                num_classes: int = 6):
        super().__init__()
        self.model = models.resnet18()
        self.model.load_state_dict(torch.load('/data/Models/image_recognition/resnet18_pretrained.pth'))
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc = Dirichlet(num_features, num_classes)
    # freeze feature extractor
    def freeze_weights(self, 
                       layers: list = [1,2,3]):
        if 1 in layers:
            for p in self.model.layer1.parameters():
                p.requires_grad = False
        if 2 in layers:
            for p in self.model.layer2.parameters():
                p.requires_grad = False
        if 3 in layers:
            for p in self.model.layer3.parameters():
                p.requires_grad = False
        if 4 in layers:
            for p in self.model.layer4.parameters():
                p.requires_grad = False
    def forward(self, 
                x: torch.tensor):
        pred = self.model(x)
        pred = self.fc(pred)
        return pred




class resnext50(nn.Module):
    def __init__(self, 
                num_classes: int = 6):
        super().__init__()
        self.model = models.resnext50_32x4d()
        self.model.load_state_dict(torch.load('/data/Models/image_recognition/resnext50_pretrained.pth'))
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc = Dirichlet(num_features, num_classes)
    # freeze feature extractor
    def freeze_weights(self, 
                       layers: list = [1,2,3]):
        if 1 in layers:
            for p in self.model.layer1.parameters():
                p.requires_grad = False
        if 2 in layers:
            for p in self.model.layer2.parameters():
                p.requires_grad = False
        if 3 in layers:
            for p in self.model.layer3.parameters():
                p.requires_grad = False
        if 4 in layers:
            for p in self.model.layer4.parameters():
                p.requires_grad = False
    def forward(self, 
                x: torch.tensor):
        pred = self.model(x)
        pred = self.fc(pred)
        return pred
