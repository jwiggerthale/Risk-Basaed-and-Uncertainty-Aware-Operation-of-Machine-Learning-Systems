
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from PIL import Image
import numpy as np
import random

# map image class to class id
class_map = {'beetle': 0, 'possum': 1, 'sunflower': 2, 'camel': 3, 'palm_tree': 4, 'forest': 5, 'squirrel': 6, 'girl': 7, 'dinosaur': 8, 'cockroach': 9, 'trout': 10, 'wolf': 11, 'elephant': 12, 'pine_tree': 13, 'caterpillar': 14, 'ray': 15, 'pear': 16, 'cloud': 17, 'bottle': 18, 'telephone': 19, 'pickup_truck': 20, 'fox': 21, 'motorcycle': 22, 'flatfish': 23, 'seal': 24, 'tractor': 25, 'leopard': 26, 'can': 27, 'turtle': 28, 'tank': 29, 'bed': 30, 'plain': 31, 'keyboard': 32, 'train': 33, 'mouse': 34, 'whale': 35, 'television': 36, 'sweet_pepper': 37, 'bowl': 38, 'oak_tree': 39, 'raccoon': 40, 'sea': 41, 'orange': 42, 'lizard': 43, 'dolphin': 44, 'rabbit': 45, 'shrew': 46, 'worm': 47, 'clock': 48, 'lobster': 49, 'rocket': 50, 'aquarium_fish': 51, 'woman': 52, 'mountain': 53, 'tulip': 54, 'hamster': 55, 'rose': 56, 'otter': 57, 'plate': 58, 'house': 59, 'skyscraper': 60, 'chimpanzee': 61, 'lamp': 62, 'bicycle': 63, 'baby': 64, 'chair': 65, 'cattle': 66, 'table': 67, 'tiger': 68, 'maple_tree': 69, 'spider': 70, 'snail': 71, 'crab': 72, 'bear': 73, 'poppy': 74, 'bee': 75, 'streetcar': 76, 'wardrobe': 77, 'mushroom': 78, 'skunk': 79, 'porcupine': 80, 'shark': 81, 'beaver': 82, 'lawn_mower': 83, 'couch': 84, 'snake': 85, 'road': 86, 'lion': 87, 'castle': 88, 'bus': 89, 'butterfly': 90, 'orchid': 91, 'boy': 92, 'bridge': 93, 'man': 94, 'crocodile': 95, 'willow_tree': 96, 'cup': 97, 'kangaroo': 98, 'apple': 99}

# super class overview
super_classes = {'aquatic mammals': [82, 44, 57, 24, 35],
                'fish': [51, 23, 15, 81, 10],
                'flowers': [91, 74, 56, 2, 54],
                'food containers': [18, 38, 27, 97, 58],
                'fruit and vegetables': [99, 78, 42, 16, 37],
                'household electrical devices': [48, 32, 62, 19, 36],
                'household furniture': [30, 65, 84, 67, 77],
                'insects': [75, 0, 90, 14, 9],
                'large carnivores': [73, 26, 87, 68, 11],
                'large man-made outdoor things': [93, 88, 59, 86, 60],
                'large natural outdoor scenes': [17, 5, 53, 31, 41],
                'large omnivores and herbivores': [3, 66, 61, 12, 98],
                'medium-sized mammals': [21, 80, 1, 40, 79],
                'non-insect invertebrates': [72, 49, 71, 70, 47],
                'people': [64, 92, 7, 94, 52],
                'reptiles': [95, 8, 43, 85, 28],
                'small mammals': [55, 34, 45, 46, 6],
                'trees': [69, 39, 4, 13, 96],
                'vehicles 1': [63, 89, 22, 20, 33],
                'vehicles 2': [83, 50, 76, 29, 25]}


# function which creates cost matrix based on super class structure
def get_cost_matrix(cost_factor: int = 2):
    class_to_superclass = {}
    for superclass, classes in super_classes.items():
        for class_id in classes:
            class_to_superclass[class_id] = superclass
    cost_matrix = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            if i == j:
                # no cost for correct classification
                cost_matrix[i, j] = 0
            elif class_to_superclass[i] == class_to_superclass[j]:
                # 1 if identical superclass
                cost_matrix[i, j] = 1
            else:
                # c for confusion across super classes
                cost_matrix[i, j] = cost_factor
    return cost_matrix


# I stroed the ims locally in a folder structure and assigned other class ids to the dataset than the default values (very bad idea)
# these values represent the mapping to get the correct class of corruptes image dataset
convert_to_c_class = {0: 7,
                        1: 64,
                        2: 82,
                        3: 15,
                        4: 56,
                        5: 33,
                        6: 80,
                        7: 35,
                        8: 29,
                        9: 24,
                        10: 91,
                        11: 97,
                        12: 31,
                        13: 59,
                        14: 18,
                        15: 67,
                        16: 57,
                        17: 23,
                        18: 9,
                        19: 86,
                        20: 58,
                        21: 34,
                        22: 48,
                        23: 32,
                        24: 72,
                        25: 89,
                        26: 42,
                        27: 16,
                        28: 93,
                        29: 85,
                        30: 5,
                        31: 60,
                        32: 39,
                        33: 90,
                        34: 50,
                        35: 95,
                        36: 87,
                        37: 83,
                        38: 10,
                        39: 52,
                        40: 66,
                        41: 71,
                        42: 53,
                        43: 44,
                        44: 30,
                        45: 65,
                        46: 74,
                        47: 99,
                        48: 22,
                        49: 45,
                        50: 69,
                        51: 1,
                        52: 98,
                        53: 49,
                        54: 92,
                        55: 36,
                        56: 70,
                        57: 55,
                        58: 61,
                        59: 37,
                        60: 76,
                        61: 21,
                        62: 40,
                        63: 8,
                        64: 2,
                        65: 20,
                        66: 19,
                        67: 84,
                        68: 88,
                        69: 47,
                        70: 79,
                        71: 77,
                        72: 26,
                        73: 3,
                        74: 62,
                        75: 6,
                        76: 81,
                        77: 94,
                        78: 51,
                        79: 75,
                        80: 63,
                        81: 73,
                        82: 4,
                        83: 41,
                        84: 25,
                        85: 78,
                        86: 68,
                        87: 43,
                        88: 17,
                        89: 13,
                        90: 14,
                        91: 54,
                        92: 11,
                        93: 12,
                        94: 46,
                        95: 27,
                        96: 96,
                        97: 28,
                        98: 38,
                        99: 0}


# reverse step for convert_to_c_class
convert_from_c_class = {7: 0,
                        64: 1,
                        82: 2,
                        15: 3,
                        56: 4,
                        33: 5,
                        80: 6,
                        35: 7,
                        29: 8,
                        24: 9,
                        91: 10,
                        97: 11,
                        31: 12,
                        59: 13,
                        18: 14,
                        67: 15,
                        57: 16,
                        23: 17,
                        9: 18,
                        86: 19,
                        58: 20,
                        34: 21,
                        48: 22,
                        32: 23,
                        72: 24,
                        89: 25,
                        42: 26,
                        16: 27,
                        93: 28,
                        85: 29,
                        5: 30,
                        60: 31,
                        39: 32,
                        90: 33,
                        50: 34,
                        95: 35,
                        87: 36,
                        83: 37,
                        10: 38,
                        52: 39,
                        66: 40,
                        71: 41,
                        53: 42,
                        44: 43,
                        30: 44,
                        65: 45,
                        74: 46,
                        99: 47,
                        22: 48,
                        45: 49,
                        69: 50,
                        1: 51,
                        98: 52,
                        49: 53,
                        92: 54,
                        36: 55,
                        70: 56,
                        55: 57,
                        61: 58,
                        37: 59,
                        76: 60,
                        21: 61,
                        40: 62,
                        8: 63,
                        2: 64,
                        20: 65,
                        19: 66,
                        84: 67,
                        88: 68,
                        47: 69,
                        79: 70,
                        77: 71,
                        26: 72,
                        3: 73,
                        62: 74,
                        6: 75,
                        81: 76,
                        94: 77,
                        51: 78,
                        75: 79,
                        63: 80,
                        73: 81,
                        4: 82,
                        41: 83,
                        25: 84,
                        78: 85,
                        68: 86,
                        43: 87,
                        17: 88,
                        13: 89,
                        14: 90,
                        54: 91,
                        11: 92,
                        12: 93,
                        46: 94,
                        27: 95,
                        96: 96,
                        28: 97,
                        38: 98,
                        0: 99}



'''
function to get train, validation and test ims from directory
train and validation ims are from original train ims
number of train ims (per class) can be defined when calling the function 
returns: train_files (list), val_files (list), test_files (list)
'''
def get_cifar_files(fp: str  = '/data/Uncertainty/cifar_modeling/benchmark_data/images_classic/cifar100', 
                    num_train_ims: int = 400):
    train_dir = f'{fp}/train'
    test_dir = f'{fp}/test'
    train_files = []
    test_files = []
    val_files = []

    for d in os.listdir(train_dir):
        sub_path = f'{train_dir}/{d}'
        new_files = [f'{sub_path}/{f}' for f in os.listdir(sub_path) if f.endswith('.png')]
        train_files.extend(new_files[:num_train_ims])
        val_files.extend(new_files[num_train_ims:])
    
    for d in os.listdir(test_dir):
        sub_path = f'{test_dir}/{d}'
        new_files = [f'{sub_path}/{f}' for f in os.listdir(sub_path) if f.endswith('.png')]
        test_files.extend(new_files)
    
    return train_files, val_files, test_files

    


'''
custom augmentation to add gradient to image
implements augmentation described in chapter 7 of the thesis for Severstal dataset
# not relevant here 
'''
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



'''
define transformatoions for trainining, validation and testing
'''
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomRotation(15),
    transforms.Resize(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])
transform_aug = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomRotation(15),
    transforms.Resize(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RandomGradient(p = 1.1),
    transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])
transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    #transforms.Normalize([0.09936217326743935, 0.09936217326743935, 0.09936217326743935],[0.15983977644971742, 0.15983977644971742, 0.15983977644971742])
    ])



'''
image dataset class 
initialize with: 
  image_files: list --> list of file paths 
  mappings: dict: --> mapping to convert from my class ids to default class ids
  transforms: transforms.Copmpose --> transformations to be applied to the images
__getitem__ returns: 
  im: torch.tensot --> image data
  label: int --> class label
  fp: str --> path to image (useful to find image when you examine certain patterns)
'''
class image_dataset_c(Dataset):
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

class image_dataset(Dataset):
    def __init__(self,  
                 image_files: list,
                 mappings: dict,
                 transforms: transforms.Compose = None):
        super().__init__()
        self.files = image_files
        self.map = mappings
        self.transforms = transforms
    def __len__(self):
        return(len(self.files))
    def __getitem__(self, index):
        fp = self.files[index]
        im = Image.open(fp).convert("RGB")
        im = np.array(im)
        label = fp.split('/')[-2]
        label = self.map[label]
        if self.transforms != None:
            im = self.transforms(im)
        return im, label, fp
            
