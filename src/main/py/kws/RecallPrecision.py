#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature


class RecallPrecision:

    def __init__(self, keywords):
        self.keywords = keywords
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
        self.y_test = []
        self.y_score = []
        self.word_dict = {}

    def reset_point(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    # Sensitivity also True Positive Rate (TPR)
    @staticmethod
    def recall(tp, fn):
        return tp / (tp + fn + (sys.float_info.epsilon if (tp + fn) == 0 else 0))

    # Positive Predictive Value (PPV)
    @staticmethod
    def precision(tp, fp):
        return tp / (tp + fp + (sys.float_info.epsilon if (tp + fp) == 0 else 0))

    @staticmethod
    def accuracy(tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn + (sys.float_info.epsilon if (tp + tn + fp + fn) == 0 else 0))

    @staticmethod
    def f1score(tp, fp, fn):
        r = RecallPrecision.recall(tp, fn)
        p = RecallPrecision.precision(tp, fp)
        return 2 * (p * r) / (p + r + (sys.float_info.epsilon if (p + r) == 0 else 0))

    def factor(self, word):
        if word in self.word_dict:
            w_factor = self.word_dict[word]
        else:
            w_factor = len(self.word_dict) + 1
            self.word_dict[word] = w_factor
        return w_factor

    def add_plot_point(self):
        self.recalls.append(self.recall(self.TP, self.FN))
        self.precisions.append(self.precision(self.TP, self.FP))
        self.reset_point()

    def add(self, word, true_word, is_call):
        if is_call:
            if word.lower() == true_word.lower():
                self.addTP(word)
            else:
                self.addFP((word, true_word))
        else:
            if true_word in self.keywords:
                self.addFN((word, true_word))
            else:
                self.addTN(word)
        # for the fancy plot: convert words to factors
        self.y_test.append(self.factor(true_word))
        self.y_score.append(self.factor(word) if is_call else (self.factor(true_word) if word != true_word else 0))

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

    def stats(self):
        recall = self.recall(self.TP_total, self.FN_total)
        precision = self.precision(self.TP_total, self.FP_total)
        accuracy = self.accuracy(self.TP_total, self.TN_total, self.FP_total, self.FN_total)
        f1score = self.f1score(self.TP_total, self.FP_total, self.FN_total)
        return str('\t Recall: {:.4f}\t Precision: {:.4f}\t Accuracy: {:.4f}\t F1score {:.4f}'
                   .format(recall, precision, accuracy, f1score))

    def plot(self, filename=''):
        plt.figure()

        # MY WAY (wrong):
        plt.plot(self.recalls, self.precisions, 'ro')
        plt.title('Recall-Precision Curve')
        # plt.show()

        # FANCY WAY source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        # not working
        # average_precision = average_precision_score(self.y_test, self.y_score)
        # print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        # precision, recall, _ = precision_recall_curve(self.y_test, self.y_score)
        #
        # # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        # step_kwargs = ({'step': 'post'}
        #                if 'step' in signature(plt.fill_between).parameters
        #                else {})
        # plt.step(recall, precision, color='b', alpha=0.2,
        #          where='post')
        # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        #
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

        if filename != '':
            plt.savefig(filename)
        plt.show()
        plt.close()
