#!/usr/bin/python

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomCrop
import numpy as np
import sys
import matplotlib.pyplot as plt

"""
TODO :
- create validation set (to be run at the very end), independent to both training and testing sets
- write validation function
- reguralize some shit 
    - add to transforms : pad + crop
"""

#we'll have to optimize parameters. this might help
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--hidden_width', default=1024, type=int)
parser.add_argument('--n_epochs', default=42, type=int)
#parser.add_argument('--dropout', default=1, type=float)
args = parser.parse_args()

#gpu if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def loadDatasets(batch_size):
    global train_loader, test_loader
    # transformations (into tensors...)
    transforms = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
    # https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
    train_dataset = MNIST(root='data', train=True, download=True, transform=transforms)
    test_dataset = MNIST(root='data', train=False, download=True, transform=transforms)
    # dataloaders
    # things that load the images into the network. Necessary since you don't want to do this by hand (= 1 at a time)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


#1 hidden layer nn
class simple_MLP(nn.Module):
    def __init__(self, hidden_width):
        super(simple_MLP, self).__init__()
        self.main = nn.Sequential(nn.Linear(28 * 28, hidden_width),  #input layer
                                  nn.ReLU(),  #activation function
                                  #nn.Dropout(dropout),
                                  nn.Linear(hidden_width, hidden_width // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_width // 2, hidden_width // 4),
                                  nn.ReLU(),
                                  nn.Linear(hidden_width // 4, 10))     #first hidden layer

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
    correct = 0
    total = 0

    for iteration, (images, labels) in enumerate(train_loader):

        #set all the gradients to 0
        optimizer.zero_grad()

        #get the output
        out = model(images)

        #calculate the loss and backpropagate
        loss = loss_function(out, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        predicted = np.argmax(out.detach(), axis=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # if iteration % 100 == 0 :
        #     print("Training iteration ", iteration, "out of ", len(train_loader.dataset)/args.batch_size, "loss = ", round(losses[-1], 2), "accuracy = ", round(100*accuracies/((iteration+1)*args.batch_size), 2), "%")

    average_loss = np.mean(losses)
    accuracy = (100 * correct / total)

    print('Train accuracy %.2f%% out of %d total' % (accuracy, total))
    return((average_loss, accuracy))


#test it
def test(model, test_loader, optimizer, loss_function) :

    #no dropout
    model.eval()

    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for iteration, (images, labels) in enumerate(test_loader):
            out = model(images)
            #calculate the loss and backpropagate
            loss = loss_function(out, labels)
            losses.append(loss.item())
            predicted = np.argmax(out.detach(), axis=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # if iteration % 100 == 0 :
                # print("Testing iteration ", iteration, "out of ", len(test_loader.dataset)/args.batch_size, "loss = ", round(loss.item(), 2), "accuracy = ", round(100*accuracies/((iteration+1)*args.batch_size), 2), "%")

    average_loss = np.mean(losses)
    accuracy = (100 * correct / total)

    print('Test accuracy %.2f%% out of %d total' % (accuracy, total))

    return((average_loss, accuracy))


def trainAndTest(batch_size, learning_rate, hidden_width, n_epochs):
    global model, loss_function, optimizer, training_losses, training_accuracies, testing_losses, testing_accuracies, testing_accuracy

    print('Run MLP on MNIST wiht batch_size=%d, learning_rate=%.4f, hidden_width=%d, n_epochs=%d ' %
          (batch_size, learning_rate, hidden_width, n_epochs))

    loadDatasets(batch_size=batch_size)

    model = simple_MLP(hidden_width=hidden_width)
    # send it to the device
    model.to(device)
    # define loss_fn and set optimizer up
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    training_losses = []
    training_accuracies = []
    testing_losses = []
    testing_accuracies = []
    # train and test
    for epoch in range(n_epochs):
        print("Epoch ", epoch)
        training_results = train(model, train_loader, optimizer, loss_function)
        training_losses.append(training_results[0])
        training_accuracies.append(training_results[1])
        testing_results = test(model, test_loader, optimizer, loss_function)
        testing_losses.append(testing_results[0])
        testing_accuracies.append(testing_results[1])

    print('accuracy results: ' + str(testing_results))
    return testing_accuracies[-1]


if __name__ == '__main__':

    print(round(trainAndTest(batch_size=args.batch_size,
                       learning_rate=args.learning_rate,
                       hidden_width=args.hidden_width,
                       n_epochs=args.n_epochs), 2),
          end='')

    # ### this could go in plotResult() function ###
    # plotting disabled for run on cluster
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.plot(np.arange(args.n_epochs), training_losses, color="blue", label="training set loss")
    # plt.plot(np.arange(args.n_epochs), testing_losses, color="red", label="test set loss")
    # plt.xlabel("training epoch")
    # plt.ylabel("loss")
    # plt.legend(loc='upper right')
    #
    # plt.subplot(1,2,2)
    # plt.plot(np.arange(args.n_epochs), training_accuracies, color="blue", label="training set acc")
    # plt.plot(np.arange(args.n_epochs), testing_accuracies, color="red", label="test set acc")
    # plt.xlabel("training epoch")
    # plt.ylabel("accuracy")
    # plt.legend(loc='upper right')
    #
    # plt.tight_layout()
    # plt.savefig("MLP_4.5_Biologists.png")
    # plt.show()
