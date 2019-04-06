#!/usr/bin/python

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix  
import numpy as np
from sklearn.model_selection import GridSearchCV
from torchvision.datasets import MNIST

def train(model, data) :
    train_labels = data[:,0]
    train_samples = data[:,1:]

    model.fit(train_samples, train_labels)

def test(model, data) :
    test_labels = data[:,0]
    test_samples = data[:,1:]

    predictions = model.predict(test_samples)

    results = sum(predictions == test_labels)/len(test_labels)

    print("report : \n", classification_report(test_labels, predictions))
    
    return(results)

def train_and_test(model, train_data, test_data) :
    train(model, train_data)
    results = test(model, test_data)

    print("model : ", model, "\nresults : ", 100*results, "%")
    
    return(results)

def find_C_linear_kernel(data):
    train_data = data[:, 1:]
    train_labels = data[:, 0]

    # set parameters for the SVC function (apply different C values)
    tuned_parameters = [{"kernel": ["linear"],
                         "C": [10**i for i in np.arange(-8, 0, 0.5)]}]
    # the chosen range for the gridsearch is sufficient for this data but to have a more general function the range should maybe be bigger
    # but a bigger range will take longer to compute, especially for the rbf model below

    # grid-search with cross-validation to optimize C
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=3)
    clf.fit(train_data, train_labels)

    # print the average accuracy during cross validation
    means = clf.cv_results_["mean_test_score"]
    parameters = clf.cv_results_["params"]
    for mean, params in zip(means, parameters):
        print("{} {:>10.3e}{:>15}{:>8.3f}".format("C:", params["C"], "mean score:", mean))

    return (clf.best_params_)

def find_C_gamma_RBF_kernel(data):
    train_data = data[:, 1:]
    train_labels = data[:, 0]
    # set parameters for the SVC function (apply different C and gamma parameters)
    tuned_parameters = [{"kernel": ["rbf"],
                         "C": [10**i for i in np.linspace(-1, 4, 6)],
                         "gamma": [10**i for i in np.linspace(-8, -5, 4)]}]
    #maybe consider smaller stepsize for the gridsearch, but it will take much longer

    # Grid-search with Cross-validation to optimize C and gamma
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=3)
    clf.fit(train_data, train_labels)

    # print the average accuracy during cross validation
    means = clf.cv_results_["mean_test_score"]
    parameters = clf.cv_results_["params"]
    for mean, params in zip(means, parameters):
        print("{} {:>10.3e}{:>10}{:>12.3e}{:>15}{:>8.3f}".format("C:", params["C"], "gamma:", params["gamma"], "mean score:", mean))

    return (clf.best_params_)


if __name__ == '__main__':
    #this part is for parameter optimization on the reduced MNIST set
    # train_data = np.loadtxt(open("data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)
    # test_data = np.loadtxt(open("data/test_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)
    # #in case the symlink doesn't work for you, you might have to use this... How stupid does this path look EH ?
    # #train = np.loadtxt(open("../../../../data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)
    #
    #
    # #find parameters for linear model reduced data set
    # print("optimizing parameter for linear model ...")
    # linear_parameters= find_C_linear_kernel(train_data)
    # print()
    # print(linear_parameters)  # output: {'C': 1e-06, 'kernel': 'linear'}
    # print()
    #
    # #find parameters for rbf model using the reduced data set
    # print("optimizing parameters for rbf model ...")
    # rbf_parameters = find_C_gamma_RBF_kernel(train_data)
    # print(rbf_parameters)   # output: {'C': 10.0, 'gamma': 1e-07, 'kernel': 'rbf'}
    #
    #
    # #applying tuned parameters for linear model on reduced Data set
    # print("testing linear model with tuned parameter")
    # linear_model = svm.SVC(kernel="linear", decision_function_shape="ovr", C=linear_parameters["C"])
    # print()
    # print(train_and_test(linear_model, train_data, test_data))
    # print()
    #
    #
    # #applying tuned parameters for rbf model on reduced Data set
    # print("testing rbf model with tuned parameters")
    # rbf_model = svm.SVC(kernel="rbf", decision_function_shape="ovr", C=rbf_parameters["C"], gamma=rbf_parameters["gamma"])
    # print(train_and_test(rbf_model, train_data, test_data))
    #
    # linear_model_better = svm.LinearSVC(penalty='l2',loss='squared_hinge', dual=False, tol=0.0001)
    # train_and_test(linear_model_better, train_data, test_data)


    #####################################################################################

    #the following is to apply the optimzed parameters on the full MNIST data set


    #load the whole MNIST data set
    train_dataset = MNIST(root='src/main/py/mlp/data', train=True, download=True, transform=False)
    training_data = train_dataset.train_data.numpy().reshape(len(train_dataset), 28*28)
    train_data_labels = train_dataset.train_labels.numpy().reshape(len(train_dataset))
    full_train_data=np.zeros((len(train_dataset), 28*28+1))
    full_train_data[:, 0]=train_data_labels
    full_train_data[:, 1:]= training_data

    test_dataset = MNIST(root='src/main/py/mlp/data', train=False, download=True, transform=False)
    testing_data = test_dataset.test_data.numpy().reshape(len(test_dataset), 28*28)
    test_data_labels = test_dataset.test_labels.numpy().reshape(len(test_dataset))
    full_test_data=np.zeros((len(test_dataset), 28*28+1))
    full_test_data[:, 0]=test_data_labels
    full_test_data[:, 1:]= testing_data

    #test linear kernel
    print("testing linear model with tuned parameter")
    linear_model = svm.SVC(kernel="linear", decision_function_shape="ovr", C=1e-06)
    print()
    print(train_and_test(linear_model, full_train_data, full_test_data))
    print()
    #test rbf kernel
    print("testing rbf model with tuned parameters")
    rbf_model = svm.SVC(kernel="rbf", decision_function_shape="ovr", C=10.0, gamma=1e-07)
    print(train_and_test(rbf_model, full_train_data, full_test_data))
