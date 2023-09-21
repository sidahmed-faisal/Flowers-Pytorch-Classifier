import os
import numpy as np
# import matplotlib.pyplot as plt
import json
import argparse

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

    parser =  argparse.ArgumentParser(description="Arguments for the model prediction script")

    parser.add_argument('--checkpoint_path', help="checkpoint file path",default='checkpoint.pth')
    parser.add_argument('--image_path', help="This is a image file that you want to classify",default='flowers/test/49/image_06213.jpg')
    parser.add_argument('--category_names', help="json file to categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="number of top k likely classes to predict, default is 5", default=5, type=int)
    parser.add_argument ('--gpu', help = "Input device you if you want to use it", type = str,default='cpu',choices=['gpu', 'cpu'])

    
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True)
        
    else:
        model = models.resnet152(pretrained=True)
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.input_units = checkpoint['input_units']
    model.output_units = checkpoint['output_units']
    model.learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    # optimizer.load_state_dict(checkpoint['optimizer']) 
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
                                                                
    pil_image = Image.open(image_path)
    transformed_image = image_transforms(pil_image)
    return transformed_image

def predict(image_path, model, topk,category_names):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)    
    
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()
    
    if args.gpu=='gpu':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    image = image.to(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(topk, dim=1)
        idx_to_class = {value:cat_to_name[key] for key, value in model.class_to_idx.items()}
        predicted_flowers = [idx_to_class[i] for i in top_classes.tolist()[0]]
        predicted_probabilities = top_ps.tolist()[0]
        classes = top_classes.tolist()[0]
        print(f'the top {topk} predicted probabilities is {predicted_probabilities} for the classes {classes} with its associated flowes names {predicted_flowers}')
#     return  predicted_probabilities , classes , predicted_flowers
        

args = get_input_args()

checkpoint_path = args.checkpoint_path
image_path = args.image_path
category_names = args.category_names
top_k = args.top_k
gpu = args.top_k



loaded_model = load_checkpoint(checkpoint_path)

image = process_image(image_path)

print(predict(image_path,loaded_model,top_k,category_names))
