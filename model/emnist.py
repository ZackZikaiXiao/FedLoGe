'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


data_path = '../data/emnist'
        # args.num_classes = 100
model = 'cnn'
trans_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
trans_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

dataset_train = datasets.EMNIST(data_path, split="letters", train=True, download=True, transform=trans_train)
dataset_test = datasets.EMNIST(data_path, split="letters", train=False, download=True, transform=trans_val)
n_train = len(dataset_train)
n_test = len(dataset_test)
y_train = np.array(dataset_train.targets)
#offcial()
print(n_train)
print(n_test)
print(n_test+n_train)

