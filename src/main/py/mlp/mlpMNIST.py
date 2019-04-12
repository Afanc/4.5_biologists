#!/usr/bin/python

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomCrop
import numpy as np
import sys
import os
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
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--hidden_width', default=1024, type=int)
parser.add_argument('--n_epochs', default=15, type=int)
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
        self.main = nn.Sequential(nn.Linear(28 * 28, hidden_width),  # input layer
                                  nn.ReLU(),  #activation function
                                  #nn.Dropout(dropout),
                                  nn.Linear(hidden_width, hidden_width // 2),  # first hidden layer
                                  nn.ReLU(),
                                  # nn.Linear(hidden_width // 2, hidden_width // 4),  # second hidden layer
                                  # nn.ReLU(),
                                  # nn.Linear(hidden_width // 4, 10))  # for 2 hidden layers
                                  nn.Linear(hidden_width // 2, 10))  # for 1 hidden layer


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

    # print('Train accuracy %.2f %% out of %d total' % (accuracy, total))
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

    # print('Test accuracy %.2f %% out of %d total' % (accuracy, total))

    return((average_loss, accuracy))


def trainAndTest(batch_size, learning_rate, hidden_width, n_epochs, delta_accuracy=0.5):
    global model, loss_function, optimizer, training_losses, training_accuracies, testing_losses, testing_accuracies, testing_accuracy

    # print('Run MLP on MNIST with batch_size=%d, learning_rate=%.4f, hidden_width=%d, 1 hidden layer, \nmax_no_epochs=%d, delta_accuracy=%.2f %% '
    #       % (batch_size, learning_rate, hidden_width, n_epochs, delta_accuracy))

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
    current_accuracy = 0

    # train and test
    epoch = 0
    for epoch in range(n_epochs):
        print("Epoch ", epoch)
        training_results = train(model, train_loader, optimizer, loss_function)
        training_losses.append(training_results[0])
        training_accuracies.append(training_results[1])
        testing_results = test(model, test_loader, optimizer, loss_function)
        testing_losses.append(testing_results[0])
        testing_accuracies.append(testing_results[1])
        accuracy = testing_results[1]
        if abs(accuracy - current_accuracy) < delta_accuracy:
            break
        current_accuracy = accuracy

    # print('Final results: average_loss:%.2f accuracy:%.2f %% after %d epochs'
    #       % (testing_results[0], testing_results[1], epoch+1))
    return (testing_accuracies[-1], epoch+1)


def plotResultPanel(x_vals, y_vals1, y_vals2, x_axis_label, y_axis_label, line_label1, \
                    line_label2, line_color1, line_color2, plot_title):
    plt.plot(np.arange(x_vals), y_vals1, color=line_color1, label=line_label1)
    plt.plot(np.arange(x_vals), y_vals2, color=line_color2, label=line_label2)
    plt.title(plot_title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(loc='right')


def plotResults():
    figure = plt.figure(figsize=(10,5))
    plt.suptitle("Batch size: " + str(args.batch_size) + ", learning rate: " + str(args.learning_rate) \
                 + ", hidden width: " + str(args.hidden_width) + ", 1 hidden layer, no. of epochs: " + str(result[1]))
    plt.subplot(1,2,1)
    plotResultPanel(epoch_number_final, training_losses, testing_losses, x_axis_label = "training epoch", \
                    y_axis_label = "loss", line_label1 = "training set", line_label2 = "test set", \
                    line_color1 = "blue", line_color2 = "red", plot_title = "Loss")
    plt.subplot(1,2,2)
    plotResultPanel(epoch_number_final, training_accuracies, testing_accuracies, x_axis_label = "training epoch", \
                    y_axis_label = "accuracy", line_label1 = "training set", line_label2 = "test set", \
                    line_color1 = "blue", line_color2 = "red", plot_title = "Accuracy")
    figure_name = "MLP_4.5_Biologists__bs_" + str(args.batch_size) + "__lr_" + str(args.learning_rate) + "__hw_" \
                  + str(args.hidden_width) + "__no of epochs_" + str(epoch_number_final) + ".png"
    path = "figures"
    try:
        os.makedirs(path)
    except FileExistsError:  # directory already exists
        pass
    full_name = os.path.join(path, figure_name)
    figure.savefig(full_name)


if __name__ == '__main__':

    result = trainAndTest(batch_size=args.batch_size,
                       learning_rate=args.learning_rate,
                       hidden_width=args.hidden_width,
                       n_epochs=args.n_epochs)
    # result[0]: testing accuracy;    result[1]: number of epochs to reach testing accuracy
    epoch_number_final = result[1]
    print(round(result[0], 2), end='')

    plotResults()