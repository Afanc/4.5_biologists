#!/usr/bin/python

import os
import csv
from wcmatch import fnmatch
import numpy as np
from scipy.spatial.distance import euclidean, cityblock
import dtw

import scan_image_features as sif


class DynTimeWrap:
    """Dynamic Time Warping train, validate, search for keywords.
        object arguments:
        - num_f: number of features
        - f_width: feature width
        - paths: data file paths
            required:
            - resized_word_images
    """
    def __init__(self, paths, numb_f=4, f_width=212):
        self.paths = paths
        self.numb_f = numb_f
        self.f_width = f_width
        self.words_and_features = []

    def train(self, train_pages, save_file_name=''):
        file_filters = []
        for page in train_pages:
            file_filters.append(page + '-*.png')
        word_images = sorted(fnmatch.filter(os.listdir(self.paths["resized_word_images"]), file_filters))

        self.words_and_features = self.get_word_features(word_images)
        if len(save_file_name) != 0:
            with open(save_file_name, 'w') as f:
                writer = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
                for feature in self.words_and_features:
                    writer.writerow(feature)
        return self.words_and_features

    def load_word_features(self, features_file_name):
        self.words_and_features = []
        with open(features_file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.words_and_features.append(row)
        return self.words_and_features

    def get_word_features(self, word_images):       # TODO buggy: fix it
        features = np.zeros(shape=(len(word_images), self.numb_f, self.f_width))
        words = []
        for i, w in enumerate(word_images):
            if w.startswith('.'):
                continue
            word_image = os.path.join(self.paths["resized_word_images"], w)
            f = sif.scan_image_features(word_image, self.numb_f, normalize_feature_matrix=True)
            features[i] = f
            words.append(w[10:-4])
            if i % 100 == 0:
                print("feature extraction, image ", i, "out of", len(word_images))
        return [[w, features[i]] for i, w in enumerate(words)]

    def spot_keywords(self, pages, keywords, result_file_name=''):
        key_features = []
        for kwf in self.words_and_features:
            if kwf[0] in keywords:
                key_features.append(kwf)

        file_filters = []
        for page in pages:
            file_filters.append(page + '-*.png')
        word_images = sorted(fnmatch.filter(os.listdir(self.paths["resized_word_images"]), file_filters))

        validate_word_features = self.get_word_features(word_images)

        spotted_words = []
        for w, wf in validate_word_features:
            for kwf in key_features:
                (d, cost_matrix, acc_cost_matrix, path) = dtw.dtw(kwf[1], wf, dist=cityblock)
                guess = d > 100
                if guess:
                    spotted_words.append((spotted_word, real_word, (d, cost_matrix, acc_cost_matrix, path)))
                spotted_word = kwf[0]
                real_word = w
                if guess and spotted_word == real_word:
                    print('good guess for: ' + real_word)
                else:
                    print('bad guess for: ' + real_word + ' miss matched with ' + spotted_word)

        if len(result_file_name) != 0:
            with open(result_file_name, 'w') as fr:
                writer = csv.writer(fr, lineterminator='\n')
                for word_stat in spotted_words:
                    writer.writerow(word_stat)

        # TODO maybe plot accuracy plot
        return

    """Dynamic Time Warping between two feature vector sequences"""
    def dyn_time_warp(word1, word2):
        d, cost_matrix, acc_cost_matrix, path = dtw.dtw(word1, word2, dist=cityblock)
        #or dtw.accelerated_dtw(...) ?
        #or euclidean ?

        return (d, cost_matrix, acc_cost_matrix, path)
