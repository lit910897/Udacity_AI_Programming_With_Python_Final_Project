from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
import json
import seaborn as sb
import argparse
#execute: train.py './flowers'
def parse():
    parser = argparse.ArgumentParser(description = "Set parser for training the network")
    parser.add_argument('data_dir', help = 'Data directory(required)', type = str)
    parser.add_argument('--save_dir',help = 'Saving directory(optional)', default = './' , type = str)
    parser.add_argument('--arch', help = 'Model architecture. Options = [vgg16, densenet121]', default='vgg16', type = str)
    parser.add_argument('--lr', help ='Learning rate', default = 0.001, type = float)
    parser.add_argument('--hidden_units', help = 'Number of classifier hidden units(as list[4096,1000]',default = [4096, 1000], type = int)
    psrser.add_argument('--epochs', help = 'Number of epochs', default = 5, type = int)
    parser.add_argument('--GPU', help = "Option to use 'GPU'(yes/no)", default = 'yes', type = str)
    parser.add_argument('--dropout', help = 'Set dropout rate', default = 0.2)
    
    args = parser.parse_args()
    
    return args

def device(args):
    if args.GPU == 'yes':
        args.device = 'cuda'
    elif arg.GPU == 'no':
        args.device = 'cpu'
    return

def access_data(args):
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    data_dir = [train_dir, valid_dir, test_dir]
    
    return data_dir

def transform_and_load(data_dir):
    train_dir, valid_dir, test_dir = data_dir
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    
    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
    data_sets = {'train':train_data, 'valid':valid_data, 'test':test_data} 
    loaders = {'train':trainloader,'valid':validloader,'test':testloader,'labels':cat_to_name}
    
    return loaders,data_sets

def build_model(args):
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        args.input_size = 25088
        
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        args.input_size = 1024
        
    #Turn off gradients for our model
    for param in model.parameters():
        param.requires_grad = False
        
    drp = arg.dropout
    hidden_size = arg.hidden_units
    output_size = 102
    classifier = nn.Sequential(nn.Linear(args.input_size,hidden_size[0]),
                              nn.ReLU(),
                              nn.Dropout(drp),
                              nn.Linear(hidden_size[0],hidden_size[1]),
                              nn.ReLU(),
                              nn.Dropout(drp),
                              nn.Linear(hidden_size[1],output_size),
                              nn.LogSoftmax(dim = 1))
    model.classifier = classifier
    return model

def set_optimizer_criterion(model,args):
    optimizer = optim.Adam(model.classifier.parameters(),lr=args.lr)
    criterion = nn.NLLLoss()
    return optimizer, criterion

def train_model(model, args, loaders, optimizer, criterion):
    model.to(args.device)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    
    #log training to file
    with open("training_log.txt", "w") as f:
        for epoch in range(epochs):
            model.train()
            for inputs, labels in loaders['train']:
                steps += 1
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                
                optimizer.zero_grad()
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    model.eval()
                    valid_loss = 0
                    accuracy = 0
                    
                    with torch.no_grad():
                        for inputs, labels in loaders['valid']:
                            inputs, labels = inputs.to(args.device), labels.to(args.device)
                            logps = model(inputs)
                            loss = criterion(logps, labels)
                            
                            valid_loss += loss.item()
                            
                            #Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    print(f"Epoch {epoch+1}/{epochs}.."
                          f"Train loss: {running_loss/print_every:.3f}.."
                          f"Validation loss: {valid_loss/len(loaders['valid']):.3f}.."
                          f"Validation accuracy: {accuracy/len(loaders['valid']):.3f}..")
                    
                    f.write(f"Epoch {epoch+1}/{epochs}.."
                          f"Train loss: {running_loss/print_every:.3f}.."
                          f"Validation loss: {valid_loss/len(loaders['valid']):.3f}.."
                          f"Validation accuracy: {accuracy/len(loaders['valid']):.3f}..")
                    
                    f.write('\n')
                    running_loss = 0
                    model.train()
    f.close()
    return model

def validate(model, args, loaders, criterion):
    model.eval()
    accuracy = 0
    test_loss = 0
    
    with torch.no_grad():
        for inputs, labels in loaders['test']:
            model.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(arg.device)
            logps = model(inputs)
            
            test_loss += criterion(logps, labels).item()
            
            #Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    print(f"Test accuracy: {accuracy/len(loaders['test']):.3f}")
    return

def save_model(model, args, optimizer, data_sets):
    model.class_to_idx = data_sets['train'].class_to_idx
    
    checkpoint = {'learning_rate': args.lr,
                  'epochs': args.epochs,
                  'batch_size': 32,
                  'model': model.cpu(),
                  'features': model.features,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    toch.save(checkpoint, str(args.save_dir + 'checkpoint.pth'))
    print('Model saved successfully')
    return

def main():
    args = parse()
    data = access_data(args)
    loaders, data_sets = transform_and_load(data)
    device(args)
    model = build_model(args)
    optimizer, criterion = set_optimizer_criterion(model, args)
    model = train_model(model, args, loaders, optimizer, criterion)
    validate(model, args, loaders, criterion)
    save_model(model, args, optimizer, data_sets)
    print(model)
    print('training finished successfully')
    
if __name__ == '__main__':
    main()
    
                          
    