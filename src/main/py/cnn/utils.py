#!/usr/bin/python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomCrop, Grayscale
import numpy as np
import torchvision
import sys
import matplotlib.pyplot as plt

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

def loadDatasetsPermutMNIST(batch_size):
    global train_loader, test_loader
    # transformations (into tensors...)
    transforms = Compose([Grayscale(num_output_channels=1), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
    train_dataset = torchvision.datasets.ImageFolder(
        root='mnist-permutated-png-format/mnist/train',
        transform=transforms
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root='mnist-permutated-png-format/mnist/test',
        transform=transforms
    )
    # dataloaders
    # things that load the images into the network. Necessary since you don't want to do this by hand (= 1 at a time)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def train(model, train_loader, optimizer, loss_function, batch_size) :

    #this does nice shit in case we dropout or other
    model.train()

    #and other things we would like to track
    losses = []
    correct = 0
    total = 0

    for iteration, (images, labels) in enumerate(train_loader):
        images.to(device)
        labels.to(device)

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

        if iteration % 100 == 0 :
            print("Training iteration ", iteration, "out of ", len(train_loader.dataset)/batch_size, "loss = ", round(losses[-1], 2), "accuracy = ", round(100*correct/total, 2), "%")

    average_loss = np.mean(losses)
    accuracy = (100 * correct / total)

    print('Train accuracy %.2f%% out of %d total' % (accuracy, total))
    return((average_loss, accuracy))


def test(model, test_loader, optimizer, loss_function, batch_size) :

    #no dropout
    model.eval()

    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for iteration, (images, labels) in enumerate(test_loader):
            images.to(device)
            labels.to(device)
            out = model(images)
            #calculate the loss and backpropagate
            loss = loss_function(out, labels)
            losses.append(loss.item())
            predicted = np.argmax(out.detach(), axis=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if iteration % 100 == 0 :
                print("Testing iteration ", iteration, "out of ", len(test_loader.dataset)/batch_size, "loss = ", round(loss.item(), 2), "accuracy = ", round(100*correct/total, 2), "%")

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

