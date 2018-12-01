import argparse
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from PIL import Image
import os
import numpy as np

####################################
# Trainig purpose functions
####################################

# For fetching command line arguments for training
def get_inp_args_train():
    parser = argparse.ArgumentParser()
    current_dir = os.getcwd()
    parser.add_argument("data_dir", type = str)
    parser.add_argument("--save_dir", type = str, default = current_dir)
    parser.add_argument("--arch", type = str, default = "densenet121")
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--hidden_units", type = int, default = 500)
    parser.add_argument("--epochs", type = int, default = 3)
    parser.add_argument("--gpu", action = "store_true")
    args = parser.parse_args()
    
    return args

# Validation pass function
def validation(model, validloader, criterion, device):
    loss = 0
    accuracy = 0
    for inputs, labels in validloader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model.forward(inputs)
        loss += criterion(outputs, labels)
        
        prob = torch.exp(outputs)
        equality = (prob.max(dim=1)[1] == labels)
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return loss, accuracy

# Saves checkpoint
def save_checkpoint(model, optimizer, epochs, save_dir, train_dataset, arch):
    checkpoint = {'input_size': model.classifier[0].in_features,
                 'output_size': model.classifier[3].out_features,
                 'hidden_layers': [model.classifier[i].out_features for i in (0,3)],
                 'classifier_state_dict': model.classifier.state_dict(),
                 'optim_state_dict': optimizer.state_dict(),
                 'drop': model.classifier[2].p,
                 'epochs': epochs,
                 'arch': arch,
                 'class_to_idx': train_dataset.class_to_idx
                 }

    torch.save(checkpoint, save_dir + '/' + 'checkpoint.pth')


#####################################
# Prediction purpose functions
#####################################

# For fetching the command line arguments for prediction
def get_inp_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type = str)
    parser.add_argument("checkpoint", type = str)
    parser.add_argument("--top_k", type = int, default = 4)
    parser.add_argument("--category_names", type = str)
    parser.add_argument("--gpu", action = "store_true")
    args = parser.parse_args()

    return args

# Loads the model
def load_model(path):
    checkpoint = torch.load(path)
    classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
                                         ("relu", nn.ReLU()),
                                         ("drop", nn.Dropout(checkpoint['drop'])),
                                         ("fc2", nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])),
                                         ("output", nn.LogSoftmax(dim=1))]))
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    model = getattr(models, checkpoint['arch'])(pretrained = True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = classifier
        
    return model

# Preprocessing the PIL image to make it fit for input to the model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    # Resizing
    old_size = pil_image.size
    new_size = []
    if old_size[0] <= old_size[1]:
        new_size.append(256)
        new_size.append(int((256*old_size[1])/old_size[0]))
    else:
        new_size.append(int((256*old_size[0])/old_size[1]))
        new_size.append(256)
    pil_image = pil_image.resize(size = new_size)
    
    # Cropping 224x224
    current_size = pil_image.size
    pil_image = pil_image.crop((int(current_size[0]/2) - 112,
                               int(current_size[1]/2) - 112,
                               int(current_size[0]/2) + 112,
                               int(current_size[1]/2) + 112))
    
    np_image = np.array(pil_image, dtype = 'float')
    np_image = np_image.T

    #Normalizing
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = np_image/255
    for i in range(np_image.shape[0]):
        np_image[i] = (np_image[i] - means[i])/std[i]
        
    return np_image

# Predicting the given image
def predict(image_path, model, topk, device):
    model.eval()
    np_image = process_image(image_path)
    
    tensor_image = torch.from_numpy(np_image)
    tensor_image = tensor_image.to(device)
    tensor_image.unsqueeze_(0)
    tensor_image = tensor_image.type(torch.FloatTensor)
    
    with torch.no_grad():
        output = model.forward(tensor_image)
        
    prob = torch.exp(output)
    top_k_values, indices = prob.topk(topk)
    top_k_values = top_k_values.numpy().squeeze()
    
    # Get class from index
    indices = indices.numpy().squeeze()
    class_to_idx = model.class_to_idx
    idx_to_class = {value:key for key, value in class_to_idx.items()}
    classes = [int(idx_to_class[idx]) for idx in indices]
    
    return top_k_values, classes
