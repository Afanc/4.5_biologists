#!/usr/bin/python

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix  
import numpy as np

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

if __name__ == '__main__':
    train_data = np.loadtxt(open("data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)
    test_data = np.loadtxt(open("data/test_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)
    #in case the symlink doesn't work for you, you might have to use this... How stupid does this path look EH ?
    #train = np.loadtxt(open("../../../../data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)     
    
    linear_model = svm.SVC(kernel="linear", decision_function_shape="ovr")
    train_and_test(linear_model, train_data, test_data)

    #linear_model_better = svm.LinearSVC(penalty='l2',loss='squared_hinge', dual=False, tol=0.0001)
    #train_and_test(linear_model_better, train_data, test_data)
