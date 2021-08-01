import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import argparse
import os

#Command Line Imports

def get_train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str, help='Data Directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='Saving Directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN architecture - vgg16 or alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_unit', type=int, default=1024, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    return parser.parse_args()

train_args = get_train_args()


#Load the data
train_dir = train_args.data_dir + '/train'
valid_dir = train_args.data_dir + '/valid'
test_dir = train_args.data_dir + '/test'

data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                   'test': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}


image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                  'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['test'])}


loaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
           'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True),
           'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)}


#Training function 
use_cuda = torch.cuda.is_available()

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, gpu, save_path):
    
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        
        train_loss = 0
        valid_loss = 0
        
        model.train()
        
        for batch_idx, (data, target) in enumerate(loaders['train']):
            
            if use_cuda and gpu:
                data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1/(batch_idx+1)) * (loss.data - train_loss))
        
        model.eval()
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            
            if use_cuda and gpu:
                data, target = data.cuda(), target.cuda()
                
            output = model(data)
            loss = criterion(output, target)
            
            valid_loss = valid_loss + ((1/(batch_idx+1)) * (loss.data - valid_loss))
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        
        print(f"Epoch: {epoch} Training loss: {train_loss:.6f} Validation loss {valid_loss:.6f} Accuracy: {100*correct/total:.6f} % ({correct} / {total})")
        
        if valid_loss < valid_loss_min:
            model.class_to_idx = image_datasets['train'].class_to_idx
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model, save_path + '/model.pt')
            
            valid_loss_min = valid_loss
        
    return model

#
if train_args.arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    in_features = model.classifier[1].in_features
else:
    model = models.vgg16(pretrained=True)
    in_features = model.classifier[0].in_features

for param in model.parameters():
    param.requires_grad = False

n = train_args.hidden_unit

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(in_features, n)),
    ('batch1', nn.BatchNorm1d(n)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(n, int(n/2))),
    ('batch2', nn.BatchNorm1d(int(n/2))),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(p=0.5)),
    ('output', nn.Linear(int(n/2), 102))
]))

model.classifier = classifier

if use_cuda and train_args.gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=train_args.learning_rate)


model = train(train_args.epochs, loaders, model, optimizer, criterion, use_cuda, train_args.gpu, train_args.save_dir)

