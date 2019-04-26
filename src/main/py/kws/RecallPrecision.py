#!/usr/bin/python

import matplotlib.pyplot as plt
import sys


class RecallPrecision:

    def __init__(self, name):
        self.name = name
        self.TP_total = 0
        self.TN_total = 0
        self.FP_total = 0
        self.FN_total = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.recalls = []
        self.precisions = []
        self.TP_words = []
        self.TN_words = []
        self.FP_words = []
        self.FN_words = []

    def reset_point(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    # Sensitivity also True Positive Rate (TPR)
    @staticmethod
    def recall(tp, fn):
        return tp / (tp + fn + (sys.float_info.epsilon if tp == fn else 0))

    # Positive Predictive Value (PPV)
    @staticmethod
    def precision(tp, fp):
        return tp / (tp + fp + (sys.float_info.epsilon if tp == fp else 0))

    def add_plot_point(self):
        self.recalls.append(self.recall(self.TP, self.FN))
        self.precisions.append(self.precision(self.TP, self.FP))
        self.reset_point()

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
        self.TP_total += 1

    def addTN(self, word):
        self.TN_words.append(word)
        self.TN += 1
        self.TN_total += 1

    def addFP(self, words):
        self.FP_words.append(words)
        self.FP += 1
        self.FP_total += 1

    def addFN(self, words):
        self.FN_words.append(words)
        self.FN += 1
        self.FN_total += 1

    def str(self):
        recall = self.recall(self.TP_total, self.FN_total)
        precision = self.precision(self.TP_total, self.FP_total)
        return 'Recall {}; Precision {}'.format(recall, precision)

    def plot(self, filename=''):
        plt.figure()
        plt.plot(self.recalls, self.precisions, 'ro', label='Recall-Precision Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if filename != '':
            plt.savefig(filename)
        plt.show()
        plt.close()
