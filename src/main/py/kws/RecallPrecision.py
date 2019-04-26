#!/usr/bin/python

import matplotlib.pyplot as plt


class RecallPrecision:
    TP = 0
    TP_words = []

    TN = 0
    TN_words = []

    FP = 0
    FP_words = []

    FN = 0
    FN_words = []

    recall = 0.
    precision = 0.
    recalls = []
    precisions = []

    def __init__(self, name):
        self.name = name

    def add_recal_precision(self):
        self.recall = self.TP / (self.TP + self.FN)   # Sensitivity also True Positive Rate (TPR)
        self.precision = self.TP / (self.TP + self.FP)  # Positive Predictive Value (PPV)
        self.recalls.append(self.recall)
        self.precisions.append(self.precision)

    def add(self, word, true_word, is_call):
        if is_call:
            if word == true_word:
                self.addTP(word)
            else:
                self.addFP((word, true_word))
        else:
            if word == true_word:
                self.addFN((word, true_word))
            else:
                self.addTN(word)

    def addTP(self, word):
        self.TP_words.append(word)
        self.TP += 1
        self.add_recal_precision()

    def addTN(self, word):
        self.TN_words.append(word)
        self.TN += 1
        # self.add_recal_precision() # so far not interested

    def addFP(self, words):
        self.FP_words.append(words)
        self.FP += 1
        self.add_recal_precision()

    def addFN(self, words):
        self.FN_words.append(words)
        self.FN += 1
        self.add_recal_precision()

    def str(self):
        return 'Recall {}; Precision {}'.format(self.recall, self.precision)

    def plot(self, filename=''):
        plt.figure()
        plt.plot(self.recall, self.precision, label='Recall-Precision Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if filename != '':
            plt.savefig(filename)
        plt.show()
        plt.close()
