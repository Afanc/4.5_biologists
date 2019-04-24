#!/usr/bin/python

import matplotlib.pyplot as plt


class RecallPrecision:
    TP = 0
    TPwords = []

    FP = 0
    FPwords = []

    FN = 0
    FNwords = []

    recall = []
    precision = []

    def __init__(self, name):
        self.name = name

    def add_recal_precision(self):
        self.recall.append(self.TP / (self.TP + self.FN))   # Sensitivity also True Positive Rate
        self.precision.append(self.TP / (self.TP + self.FP))  # Positive Predictive Power

    def addTP(self, word):
        self.TPwords.append(word)
        self.TP += 1
        self.add_recal_precision()

    def addFP(self, word):
        self.FPwords.append(word)
        self.FP += 1
        self.add_recal_precision()

    def addFN(self, word):
        self.FNwords.append(word)
        self.FN += 1
        self.add_recal_precision()

    def plot(self, filename=''):
        plt.figure()
        plt.plot(self.recall, self.precision, label='Recall-Precision Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if filename != '':
            plt.savefig(filename)
        plt.show()
        plt.close()
