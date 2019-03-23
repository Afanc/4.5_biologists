#!/usr/bin/python

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV



def train(model, data):
    train_labels = data[:, 0]
    train_samples = data[:, 1:]

    model.fit(train_samples, train_labels)


def test(model, data):
    test_labels = data[:, 0]
    test_samples = data[:, 1:]

    predictions = model.predict(test_samples)

    results = sum(predictions == test_labels) / len(test_labels)

    print("report : \n", classification_report(test_labels, predictions))
    return (results)


def train_and_test(model, train_data, test_data):
    train(model, train_data)
    results = test(model, test_data)

    #print("model : ", model, "\nresults : ", 100 * results, "%")

    return (results)

#def find_C(model, data):


if __name__ == '__main__':
    train_data = np.loadtxt(open("data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype=np.uint16)
    test_data = np.loadtxt(open("data/test_sub.csv", "rb"), delimiter=",", skiprows=0, dtype=np.uint16)
    # in case the symlink doesn't work for you, you might have to use this... How stupid does this path look EH ?
    # train = np.loadtxt(open("../../../../data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)

    # linear Kernel
    #"""
    # set parameters for the SVC function (apply different C values)
    tuned_parameters = [{"kernel": ["linear"], 
                         "C": [10**i for i in np.arange(-8, 0, 0.5)]}]
    
    # grid-search with cross-validation
    # (finding the C value resulting in the best accuracy)
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=3)
    clf.fit(train_data[:, 1:], train_data[:, 0])
    print("Kernel: linear")
    print("C:", clf.best_params_["C"])
    print("Score:",  clf.best_score_)   # c = 10e-6
    #"""

    ###################################################################
    # RBF
    # set parameters for the SVC function (apply different C values)
    #"""
    tuned_parameters = [{"kernel": ["rbf"],
                         "C": [10**i for i in np.linspace(-1, 4, 6)],
                         "gamma": [10**i for i in np.linspace(-8, -5, 4)]}]

    # Grid-search with Cross-validation
    # (finding the C value resulting in the best accuracy)
    print("Grid search ...")
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=3)
    clf.fit(train_data[:, 1:], train_data[:,0])
    print("Kernel: RBF")
    print("C:", clf.best_params_["C"])
    print("gamma:", clf.best_params_["gamma"])
    print("Score:", clf.best_score_)
    #"""

    """
    rbf_model = svm.SVC(kernel="rbf", decision_function_shape="ovr", C=10, gamma=10**-7)
    print(train_and_test(rbf_model, train_data, test_data))

    linear_model = svm.SVC(kernel="linear", decision_function_shape="ovr", C=10**-6)
    print(train_and_test(linear_model, train_data, test_data))
    """

    # linear_model_better = svm.LinearSVC(penalty='l2',loss='squared_hinge', dual=False, tol=0.0001)
    # train_and_test(linear_model_better, train_data, test_data)

