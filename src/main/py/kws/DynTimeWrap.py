#!/usr/bin/python

import os
import csv
import pickle
from wcmatch import fnmatch
import numpy as np
from scipy.spatial.distance import euclidean, cityblock
import dtw

import scan_image_features as sif
from RecallPrecision import RecallPrecision

class DynTimeWrap:
    """Dynamic Time Warping train, validate, search for keywords.
        object arguments:
        - numb_f: number of features
        - f_width: feature width
        - spot_threshold: spot distance threshold for the Dynamic Time Wrapping
        - paths: data file paths
            required:
            - resized_word_images
    """
    def __init__(self, paths, numb_f=4, f_width=212, spot_threshold=6):
        self.paths = paths
        self.numb_f = numb_f
        self.f_width = f_width
        self.words_features = []
        self.spot_threshold = spot_threshold
        self.rp = RecallPrecision('dtw')

    def train(self, train_pages, save_file_name=''):
        file_filters = []
        for page in train_pages:
            file_filters.append(page + '-*.png')
        word_images = sorted(fnmatch.filter(os.listdir(self.paths["resized_word_images"]), file_filters))

        self.words_features = self.get_word_features(word_images)
        if len(save_file_name) != 0:
            self.save_word_features(self.words_features, save_file_name)
        return self.words_features

    def save_word_features(self, words_features, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(words_features, f)
            # fieldnames = ['word', 'features']
            # writer = csv.DictWriter(f, fieldnames=fieldnames)
            # writer.writeheader()
            # # writer = csv.writer(f, lineterminator='\n', escapechar='\\')
            # for feature in words_features:
            #     writer.writerow({'word': feature[0], 'features': feature[1]})
            #     # writer.writerow(feature)

    def load_word_features(self, features_file_name):
        self.words_features = []
        with open(features_file_name, 'rb') as f:
            self.words_features = pickle.load(f)
            # # reader = csv.reader(f, lineterminator='\n', escapechar='\\')
            # reader = csv.DictReader(f)
            # for row in reader:
            #     features = list(zip(*row['features'].replace('\n', '')))
            #     self.words_features.append([row['word'], features])
            #     # self.words_features.append([row[0], row[1].replace('\n', '')])
            #     # self.words_features.append(row)
        return self.words_features

    def get_word_features(self, word_images):
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
        for kwf in self.words_features:
            if kwf[0] in keywords:
                key_features.append(kwf)

        file_filters = []
        for page in pages:
            file_filters.append(page + '-*.png')
        word_images = sorted(fnmatch.filter(os.listdir(self.paths["resized_word_images"]), file_filters))

        validate_word_features = self.get_word_features(word_images)

        spotted_words = []
        count_good_spots = 0
        count_bad_spots = 0
        for w, wf in validate_word_features:
            for kwf in key_features:
                spotted_word = kwf[0]
                real_word = w
                (d, cost_matrix, acc_cost_matrix, path) = dtw.dtw(kwf[1], wf, dist=euclidean)
                spot = d < self.spot_threshold  # TODO find the better threshold, need more features
                self.rp.add(spotted_word, real_word, spot)
                if spot:
                    spotted_words.append((spotted_word, real_word, (d, cost_matrix, acc_cost_matrix, path)))
                    if spotted_word == real_word:
                        count_good_spots += 1
                    else:
                        count_bad_spots += 1
                elif spotted_word == real_word:
                    print("could spot FN:")
                    print((w, d, cost_matrix, acc_cost_matrix, path))

        if len(result_file_name) != 0:
            with open(result_file_name, 'w') as fr:
                writer = csv.writer(fr, lineterminator='\n')
                for word_stat in spotted_words:
                    writer.writerow(word_stat)
        print("FINAL STATS:")
        print("\t Good spots TP: %d " % count_good_spots)
        print("\t Bad spots FP: %d " % count_bad_spots)
        total = len(validate_word_features)
        print("\t Total %d " % total)
        print("\t Accuracy: %d " % self.rp.str())
        print("\t Report: %s " % self.rp.str())
        # TODO maybe plot accuracy plot
        return

    """Dynamic Time Warping between two feature vector sequences"""
    def dyn_time_warp(word1, word2):
        d, cost_matrix, acc_cost_matrix, path = dtw.dtw(word1, word2, dist=cityblock)
        #or dtw.accelerated_dtw(...) ?
        #or euclidean ?

        return (d, cost_matrix, acc_cost_matrix, path)
