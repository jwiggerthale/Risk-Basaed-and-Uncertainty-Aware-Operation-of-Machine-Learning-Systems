'''
This script implements different utils to get Severstal datasets

'''


from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import pandas as pd
from PIL import Image
import numpy as np
import random



# convert label to number (class category)
label_to_num = {'Scratches': 0, 
                'Rolled': 1, 
                'Pitted': 2, 
                'Patches': 3, 
                'Inclusion': 4, 
                'Crazing': 5}



# class for random gradient augmentation - not used in course of the thesis; computationally heavy
class RandomGradient:
    def __init__(self, p=0.5, max_alpha=0.5):
        self.p = p
        self.max_alpha = max_alpha

    def __call__(self, img_tensor):
        if random.random() >= self.p:
            return img_tensor
        C, H, W = img_tensor.shape
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        a = random.uniform(0.0, self.max_alpha)
        # Horizontal oder vertikal
        if random.random() < 0.5:
            # Horizontal
            if random.random() < 0.5:
                ramp = torch.linspace(1.0 - a, 1.0 + a, steps=W).view(1, 1, W).expand(1, H, W)
            else:
                ramp = torch.linspace(1.0 + a, 1.0 - a, steps=W).view(1, 1, W).expand(1, H, W)
        else:
            # Vertikal
            if random.random() < 0.5:
                ramp = torch.linspace(1.0 - a, 1.0 + a, steps=H).view(1, H, 1).expand(1, H, W)
            else:
                ramp = torch.linspace(1.0 + a, 1.0 - a, steps=H).view(1, H, 1).expand(1, H, W)
        out = img_tensor * ramp
        return torch.clamp(out, min_val, max_val)



# augmentation for random gamma correction as introduced in Section 4.4 of the thesis
class RandomGamma:
    def __init__(self, gamma_range=(0.5, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return F.adjust_gamma(img, gamma)
        return img



transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])



transform_aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ColorJitter(
        brightness=0.6,   # 0.0â€“0.6 Variation um die Originalhelligkeit
        contrast=0.6,
        saturation=0.4,
        hue=0.05
    ),
    RandomGamma(gamma_range=(0.4, 2.5), p=0.8), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #RandomGradient(p = 1.1),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])


transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])


# class to add random gaussian noise - not used in course of the thesis
class RandomGaussianNoise:
    def __init__(self, sigma_range=(0.0, 0.1), p=0.5):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            noise = torch.randn_like(img) * sigma
            img = img + noise
            img = torch.clamp(img, 0.0, 1.0)
        return img



import cv2
from PIL import Image
from torch.utils.data import Dataset






'''
This class implements an image dataset based on numpy array
Initialize with: 
    ims: np.array --> array of images
    labels: np.array --> array of labels 
    transforms: transforms.Compose --> transformations to be applied to images
'''
class image_dataset(Dataset):
    def __init__(self, 
                 ims: np.array,
                 labels: np.array,
                 transforms: transforms.Compose = None):
        super().__init__()
        self.ims = ims
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return(len(self.ims))

    def __getitem__(self, index):
        im = self.ims[index]
        cat = self.labels[index]
        if self.transforms != None:
            im = self.transforms(im)
        if im.shape[1] == 1:
            im = im.repeat(1, 3, 1, 1)
        return im, cat
    
    
'''
This class implements a dataset based on a pandas data frame
initialize with: 
    image_files: pd.DataFrame --> df with columnsn ['File', 'Cat']
    transforms: transforms.Compose --> transformations to be applied to images

Note: getitem method returns im, cat and fp 
    --> fp can be used to quickly identify images with certain properties
'''
class image_dataset_pd(Dataset):
    def __init__(self, 
                 image_files: pd.DataFrame,
                 transforms: transforms.Compose = None):
        super().__init__()
        self.files = image_files
        self.transforms = transforms

    def __len__(self):
        return(len(self.files))

    def __getitem__(self, index):
        fp = self.files.at[index, 'File']
        fp = fp.replace('./data', '/data/HiL_XAI/SteelLocation/servestal/data')
        cat = self.files.at[index, 'Cat']
        fp = fp.replace('.json', '').replace('ann', 'img')
        im = Image.open(fp).convert("L")
        # Convert to RGB
        im = Image.merge("RGB", (im, im, im))
        im = np.array(im)
        if self.transforms != None:
            im = self.transforms(im)
        return im, cat, fp
            
