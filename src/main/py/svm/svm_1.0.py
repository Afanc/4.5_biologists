#!/usr/bin/python

from sklearn import svm
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
    
    return(results)

def train_and_test(model, train_data, test_data) :
    train(linear_model, train_data)
    results = test(linear_model, test_data)

    print("model : ", model, "\nresults : ", results, "%")
    
    return(results)

if __name__ == '__main__':
    train_data = np.loadtxt(open("data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)
    test_data = np.loadtxt(open("data/test_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)
    #in case the symlink doesn't work for you, you might have to use this... How stupid does this path look EH ?
    #train = np.loadtxt(open("../../../../data/train_sub.csv", "rb"), delimiter=",", skiprows=0, dtype = np.uint16)     

    linear_model = svm.SVC(kernel="linear", decision_function_shape="ovr") #max_iter
    train_and_test(linear_model, train_data, test_data)

