import argparse
import numpy as np
import torch
import torchvision
import os
import time

from collections import OrderedDict
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms

def main():
    args = get_arguments()
    data_dir = args.data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    dropout=0.5
    hidden_layer1 = 120
    lr = 0.001
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(1024, hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        model.cuda()
    model = model
    optimizer = optimizer
    training(trainloader=trainloader, validloader=validloader, optimizer=optimizer, criterion=criterion, model=model,epochs= args.epochs, print_every=3)
    check_accuracy_on_test(model, testloader)
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'hidden_layer1':120,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx,
                'model':model,
                'classifier': classifier,
                'optimizer': optimizer.state_dict()},
                'checkpoint.pth')


def training(trainloader, validloader, optimizer, criterion, model, print_every, epochs, steps=0):
    loss_show=[]
    model.to('cuda')
    since = time.time()
    count = 0
    print("Started The Training: ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs,labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                accuracy=0
                validation_loss = 0
                for ii, (inputs2,labels2) in enumerate(validloader):
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        validation_loss += criterion(outputs, labels2).data[0]
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                accuracy = accuracy / len(validloader)
                count += 1
                print("{}. Epoch: {}/{}\n -------------------\n".format(count, e+1, epochs),
                      "Training Loss: {:.4f}\n".format(running_loss/print_every),
                      "Validation Loss: {:.4f}\n".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.4f}\n".format(accuracy))
                running_loss = 0
    print("Finished")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def check_accuracy_on_test(model,testloader):    
    correct = 0
    total = 0
    model.to('cuda:0')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", action="store", dest="save_dir", default="." , help = "Set directory to save checkpoints")
    parser.add_argument("--model", action="store", dest="model", default="densenet121" , help = "The architechture is already set to densenet121")
    parser.add_argument("--learning_rate", action="store", dest="lr", default=0.001 , help = "Set learning rate")
    parser.add_argument("--hidden_units", action="store", dest="hidden_units", default=512 , help = "Set number of hidden units")
    parser.add_argument("--epochs", action="store", dest="epochs", default=10 , help = "Set number of epochs")
    parser.add_argument("--gpu", action="store_true", dest="cuda", default=False , help = "Use CUDA for training")
    parser.add_argument('data_path', action="store")
    return parser.parse_args()

main()