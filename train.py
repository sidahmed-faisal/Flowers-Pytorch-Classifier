import os
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import json

import torch
from torch import nn , optim
import torch.nn.functional as F
from torch.utils import data

import torchvision
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

from collections import OrderedDict
from PIL import Image

def get_input_args():

    parser =  argparse.ArgumentParser(description="Arguments for the model training script")

    parser.add_argument('--dir',type=str, help='directory for the images',default='/flowers')
    parser.add_argument('--arch',type=str, help='choose training model architecture',default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--save_dir',type=str, help='directory for model checkpoint',default='checkpoint.pth')
    parser.add_argument ('--learning_rate', help = 'learning rate', type = float, default = 0.001)
    parser.add_argument ('--hidden_units', help = 'hidden units ', type = int, default = 1000)
    parser.add_argument ('--epochs', help = 'number of epochs', type = int, default = 5)
    parser.add_argument ('--GPU', help = "Input GPU if you want to use it", type = str)

    return parser.parse_args()

def train_transform(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    train_dataset = ImageFolder(train_dir,transform=train_transforms)
    return train_dataset

def valid_transform(valid_dir):
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])
    valid_dataset = ImageFolder(valid_dir,transform=validation_transforms)
    return valid_dataset
def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])
    test_dataset =  ImageFolder(test_dir,test_transforms)
    return test_dataset

def train_loader(train_dataset,batch_size=64,shuffle=True):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader
def valid_loader(valid_dataset,batch_size=32):
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size)
    return valid_dataloader
def test_loader(test_dataset,batch_size=32):
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)
    return test_dataloader

# print(current_dir_path = os.getcwd())
current_dir_path= os.getcwd()
args = get_input_args()
data_dir = args.dir
train_dir = current_dir_path+data_dir + '/train'
valid_dir = current_dir_path+data_dir + '/valid'
test_dir = current_dir_path+data_dir + '/test'
print(train_dir)
print(valid_dir)
print(test_dir)


train_dataset = train_transform(train_dir=train_dir)
valid_dataset = valid_transform(valid_dir=valid_dir)
test_dataset = test_transform(test_dir=test_dir)

train_dataloader = train_loader(train_dataset=train_dataset,batch_size=64,shuffle=True)
valid_dataloader = valid_loader(valid_dataset=valid_dataset,batch_size=32)
test_dataloader = test_loader(test_dataset=test_dataset,batch_size=32)