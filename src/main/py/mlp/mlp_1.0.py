#!/usr/bin/python

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt

"""
TODO :
- create validation set (to be run at the very end), independent to both training and testing sets
- write validation function
- correctly track and plot losses and accuracies (matplotlib). check if it's do-able "live" ?
- reguralize some shit 
    - add to transforms : pad + crop
    - maybe add dropout ? or keep it for the cnn
- maybe, just maybe, test other loss functions
- and finally optimize parameters
    - write a bash file that runs this thing with variable batch_sizes, lr, hidden_w
    - and find the right epoch to stop training and avoid overfitting (graphs...)

"""

#we'll have to optimize parameters. this might help
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--hidden_width', default=128, type=int)
parser.add_argument('--n_epochs', default=10, type=int)
args = parser.parse_args()

#gpu if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#transformations (into tensors...)
transforms = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

#https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
train_dataset = MNIST(root='data', train=True, download=True, transform=transforms)
test_dataset = MNIST(root='data', train=False, download=True, transform=transforms)

#dataloaders
#things that load the images into the network. Necessary since you don't want to do this by hand (= 1 at a time)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

#1 hidden layer nn
class simple_MLP(nn.Module):
    def __init__(self):
        super(simple_MLP, self).__init__()
        self.main = nn.Sequential(nn.Linear(28*28, args.hidden_width),   #input layer
                                  nn.ReLU(),              #activation function
                                  nn.Linear(args.hidden_width, 10))     #first hidden layer

    #what does it do
    def forward(self,x) :
        flat = x.view(x.size(0), 28*28)  #we flatten the image (channels...)
        out = self.main(flat)             #pass it through the layers
        return(out)

#train it
def train(model, train_loader, optimizer, loss_function) :

    #this does nice shit in case we dropout or other
    model.train()

    #and other things we would like to track
    losses = []
    accuracies = 0

    for iteration, (images, labels) in enumerate(train_loader) :

        #set all the gradients to 0
        optimizer.zero_grad()

        #get the output
        out = model(images)

        #calculate the loss and backpropagate
        loss = loss_function(out, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies += (np.argmax(out.detach(), axis=1) == labels).sum().item()

        if iteration % 100 == 0 :
            print("Training iteration ", iteration, "out of ", len(train_loader.dataset)/args.batch_size, "loss = ", losses[-1], "accuracy = ", 100*accuracies/((iteration+1)*args.batch_size), "%")

    average_loss = np.mean(losses)
    average_accuracy = accuracies / len(train_dataset)

    print(100*average_accuracy, "%")

    return((average_loss, average_accuracy))

#test it
def test(model, test_loader, optimizer, loss_function) :

    #no dropout
    model.eval()

    losses = []
    accuracies = 0

    for iteration, (images, labels) in enumerate(test_loader) :

        #get the output
        out = model(images)

        #calculate the loss and backpropagate
        loss = loss_function(out, labels)

        losses.append(loss.item())
        accuracies += (np.argmax(out.detach(), axis=1) == labels).sum().item()

        if iteration % 100 == 0 :
            print("Testing iteration ", iteration, "out of ", len(test_loader.dataset)/args.batch_size, "loss = ", loss.item(), "accuracy = ", 100*accuracies/((iteration+1)*args.batch_size), "%")

    # output loss and accuracy

    average_loss = np.mean(losses)
    average_accuracy = accuracies / len(train_dataset)

    print(100*average_accuracy, "%")

    return((average_loss, average_accuracy))

if __name__ == '__main__' :

    model = simple_MLP()
    #send it to the device
    model.to(device)

    #define loss_fn and set optimizer up
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    training_losses = []
    training_accuracies = []

    testing_losses = []
    testing_accuracies = []


    #train and test
    for epoch in range(args.n_epochs) :
        print("Epoch ", epoch)

        training_results = train(model, train_loader, optimizer, loss_function)
        training_losses.append(training_results[0])
        training_accuracies.append(training_results[1])

        testing_results = test(model, train_loader, optimizer, loss_function)
        testing_losses.append(testing_results[0])
        testing_accuracies.append(testing_results[1])


    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(args.n_epochs), training_losses, color="blue", label="train loss")
    plt.plot(np.arange(args.n_epochs), testing_losses, color="red", label="test loss")
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(np.arange(args.n_epochs), training_accuracies, color="blue", label="train acc")
    plt.plot(np.arange(args.n_epochs), testing_accuracies, color="red", label="test acc")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
