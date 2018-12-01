import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
import myfunctions
import sys

# Get the command line arguments
args = myfunctions.get_inp_args_train()

data_directory = args.data_dir
train_dir = data_directory + '/train'
valid_dir = data_directory + '/valid'
test_dir = data_directory + '/test'

# Transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)

# Using the image datasets and the trainforms, defining the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 32, shuffle = True)

#Getting model arch. & freezing params
arch = args.arch
model = getattr(models, arch)(pretrained = True)
for param in model.parameters():
    param.requires_grad = False

# Getting the input size of the model
input_size = 0
if arch.startswith("densenet"):
    input_size = model.classifier.in_features
elif arch.startswith("vgg"):
    input_size = model.classifier[0].in_features
else:
    sys.exit("Invalid arch param.")    

hidden_units = args.hidden_units
    
classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(input_size, hidden_units)),
                                         ("relu", nn.ReLU()),
                                         ("drop", nn.Dropout(0.3)),
                                         ("fc2", nn.Linear(hidden_units, 102)),
                                         ("output", nn.LogSoftmax(dim=1))]))

model.classifier = classifier
    
criterion = nn.NLLLoss()
learn_rate = args.learning_rate
optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)

gpu =  args.gpu
device = torch.device("cuda:0" if (gpu and torch.cuda.is_available()) else "cpu")
if (gpu and not torch.cuda.is_available()):
    sys.exit("GPU not available!")
  
# Training 
model.to(device)
print_every = 40
steps = 0
epochs = args.epochs
for e in range(epochs):
    model.train()
    
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            
            # Evaluation mode
            model.eval()
        
            # Turning gradients off
            with torch.no_grad():
                valid_loss, valid_accuracy = myfunctions.validation(model, validloader, criterion, device)
                
            print("Epoch: {}/{}... Training Loss: {:.3f}... Validation Loss: {:.3f} ... Validation Accuracy: {:.2f}"
                  .format(e+1, epochs, running_loss/print_every, valid_loss/len(validloader), valid_accuracy/len(validloader)))
            
            running_loss = 0
            
            # Back to training mode
            model.train()

print("\nTraining Successful.")

save_dir = args.save_dir
myfunctions.save_checkpoint(model, optimizer, epochs, save_dir, train_dataset, arch)

print("\nCheckpoint saved.")
