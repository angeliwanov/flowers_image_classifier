import argparse
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image

use_cuda = torch.cuda.is_available()

def get_predict_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', type=str, help='Path to file')
    parser.add_argument('checkpoint', type=str, help='Saved model') 
    parser.add_argument('--top_k', type=int, default=5, help='Top classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='File with category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    return parser.parse_args()
  
predict_args = get_predict_args()

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

def load_checkpoint(filepath, use_cuda, gpu):
    model = None
    criterion = None
    last_epoch = None
    
    try:
        checkpoint = torch.load(filepath + '/model.pt', map_location=lambda storage, loc:storage)

        model_arch = checkpoint['arch']
        learning_rate = checkpoint['learning_rate']
        num_of_hidden_units = checkpoint['num_of_hidden_units']
        num_of_categories = checkpoint['num_of_categories']

        model = initialize_model(model_arch, num_of_categories, num_of_hidden_units)

        if model is not None:
            if use_cuda and gpu:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            model.to(device)

            optimizer = optim.Adam(model.fc.parameters(), learning_rate)
            criterion = checkpoint['loss']
            model.load_state_dict(checkpoint['state_dict'])
            model.class_to_idx = checkpoint['class_to_idx']
            optimizer.load_state_dict(checkpoint['optimizer'])
            last_epoch = checkpoint['epoch']
    except:
        print('Failed loading checkpoint')
    
    return model, criterion, last_epoch


model, criterion, last_epoch = load_checkpoint(predict_args.checkpoint, use_cuda, predict_args.gpu)


with open(predict_args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    pil_image.thumbnail((256, 256))
    cropped_image = pil_image.crop((16, 16, 240, 240))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    
    transform =  transforms.Compose([transforms.ToTensor(),normalize])
    normalized_image = transform(cropped_image)

    return normalized_image

def predict(image_path, model, use_cuda, gpu, n_topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    if use_cuda and gpu:
        print('Using GPU')
        model = model.cuda()
        image = process_image(image_path).unsqueeze_(0).cuda()
        output = model(image)
    else:
        print('Using CPU')
        output = model(process_image(image_path).unsqueeze_(0))
    
    output = torch.exp(output)
    total = torch.sum(output)
    result = torch.div(output, total)
    probs, classes = result.topk(n_topk)
    
    return probs.tolist()[0], classes.tolist()[0]

probs, classes = predict(predict_args.input, model, use_cuda, predict_args.gpu, predict_args.top_k)
names = [cat_to_name[k] for k, v in model.class_to_idx.items() if v in classes]


for prob, name in zip(probs,names):
    print(f"Class {name} with probability of {prob}")