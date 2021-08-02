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
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN architecture - vgg16 or alexnet or resnet50')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_unit', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
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
    train_acc_log = []
    valid_acc_log = []
    train_loss_log = []
    valid_loss_log = []
    
    for epoch in range(1, n_epochs+1):
        
        train_loss = 0
        train_corr = 0
        train_total = 0
        
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
            
            pred = output.data.max(1, keepdim=True)[1]
            train_corr += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
        
        valid_loss = 0
        valid_corr = 0
        valid_total = 0
        
        model.eval()
        
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            
            if use_cuda and gpu:
                data, target = data.cuda(), target.cuda()
                
            output = model(data)
            loss = criterion(output, target)
            
            valid_loss = valid_loss + ((1/(batch_idx+1)) * (loss.data - valid_loss))
            
            pred = output.data.max(1, keepdim=True)[1]
            valid_corr += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            valid_total += data.size(0)
        
        print(f"Epoch: {epoch} Training loss: {train_loss:.6f} Validation loss {valid_loss:.6f}")
        print(f"Training Accuracy: {100*train_corr/train_total:.6f} % ({train_corr} / {train_total}) Validation Accuracy: {100*valid_corr/valid_total:.6f} % ({valid_corr} / {valid_total})")
        
        train_loss_log.append(train_loss)
        valid_loss_log.append(valid_loss)
        train_acc_log.append(100*train_corr/train_total)
        valid_acc_log.append(100*valid_corr/valid_total)
        
        
        if valid_loss < valid_loss_min:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path+'/model.pt')
            valid_loss_min = valid_loss
        
        
    return model

#Initializing model function
def initialize_model(model_arch, num_of_categories, num_of_hidden_units=None):
    model_ft = None
    
    arch_to_hidden_units = {'resnet50': 512, 'vgg16':512, 'alexnet':512}
    
    if model_arch == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
    elif model_arch == 'vgg16':
        model_ft = models.vgg16(pretrained=True)
    elif model_arch == 'alexnet':
        model_ft = models.alexnet(pretrained=True)
    
    if model_ft is not None:
        
        if num_of_hidden_units is None:
            num_of_hidden_units = arch_to_hidden_units[model_arch]
            
        for param in model_ft.parameters():
            param.requires_grad = False
        
        model_last_child = list(model_ft.children())[-1]
        
        if (isinstance(model_last_child, nn.modules.linear.Linear)):
            in_features = model_last_child.in_features
        else:
            list_of_children = list(model_last_child.children())
            for i in range(len(list_of_children)):
                if (isinstance(list_of_children[i], nn.modules.linear.Linear)):
                    in_features = model_last_child[i].in_features
                    break
        
        my_classifier = nn.Sequential(nn.Linear(in_features, num_of_hidden_units),
                                     nn.BatchNorm1d(num_of_hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(num_of_hidden_units, num_of_categories))
        
        model_ft.fc = model_ft.classifier = my_classifier
        
    return model_ft

#Initialize model
model = initialize_model(train_args.arch, 102, train_args.hidden_unit)

if use_cuda:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

model = train(train_args.epochs, loaders, model, optimizer, criterion, use_cuda, train_args.gpu, 'min_valid_loss') 


def save_checkpoint(model, optimizer, criterion, arch, learning_rate, num_of_hidden_units, 
                   num_of_categories, class_to_idx, save_path, last_epoch):
    
    checkpoint = {'epoch': last_epoch,
                  'class_to_idx': class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'loss': criterion,
                  'arch': arch,
                  'learning_rate': learning_rate,
                  'num_of_hidden_units': num_of_hidden_units,
                  'num_of_categories': num_of_categories
    }
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    torch.save(checkpoint, save_path+'/model.pt')
  
save_checkpoint(model, optimizer, criterion, train_args.arch, train_args.learning_rate, train_args.hidden_unit, 102, image_datasets['train'].class_to_idx, train_args.save_dir, train_args.epochs)