import argparse
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def get_predict_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', type=str, help='Path to file')
    parser.add_argument('checkpoint', type=str, help='Saved model') 
    parser.add_argument('--top_k', type=int, default=5, help='Top classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='File with category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    return parser.parse_args()
  
predict_args = get_predict_args()

model = torch.load(predict_args.checkpoint + '/model.pt')

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

    
    transform =  transforms.Compose([transforms.ToTensor(),normalize,])
    normalized_image = transform(cropped_image)

    return normalized_image

def predict(image_path, model, n_topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    output = model(process_image(image_path).unsqueeze_(0).cuda())
    probs, classes = output.topk(n_topk)
    
    return probs.tolist()[0], classes.tolist()[0]

probs, classes = predict(predict_args.input, model, predict_args.top_k)
names = [cat_to_name[k] for k, v in model.class_to_idx.items() if v in classes]


for prob, name in zip(probs,names):
    print(f"Class {name} with probability of {prob}")