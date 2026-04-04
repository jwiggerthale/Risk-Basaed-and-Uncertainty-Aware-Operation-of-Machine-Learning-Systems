
'''
this scrtip implements cost matrix for CIFAR-100 dataset
'''

import numpy as np


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

def get_cost_matrix_cifar(cost_factor: int = 2):
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
