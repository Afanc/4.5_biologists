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
    def __init__(self, paths, numb_f=9, f_width=212, spot_threshold=6.):   # TODO find the better threshold
        self.paths = paths
        self.numb_f = numb_f
        self.f_width = f_width
        self.words_features = []
        self.learned_words = set()
        self.spot_threshold = spot_threshold
        self.rp = RecallPrecision('dtw')

    def train(self, train_pages, save_file_name=''):
        file_filters = []
        for page in train_pages:
            file_filters.append(str(page) + '-*.png')
        word_images = sorted(fnmatch.filter(os.listdir(self.paths["resized_word_images"]), file_filters))

        self.words_features = self.get_word_features(word_images)
        if len(save_file_name) != 0:
            self.save_word_features(self.words_features, save_file_name)

        for wf in self.words_features:
            self.learned_words.add(wf[0])
        return self.words_features

    def save_word_features(self, words_features, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(words_features, f)

    @staticmethod
    def load_word_features(features_file_name):
        words_features = []
        with open(features_file_name, 'rb') as f:
            words_features = pickle.load(f)
        return words_features

    def train_word_features(self, features_file_name):
        self.words_features = self.load_word_features(features_file_name)
        for wf in self.words_features:
            self.learned_words.add(wf[0])
        return self.words_features

    def get_word_features(self, word_images):
        features = np.zeros(shape=(len(word_images), self.numb_f, self.f_width))
        words = []
        locations = []
        for i, w in enumerate(word_images):
            if w.startswith('.'):
                continue
            word_image = os.path.join(self.paths["resized_word_images"], w)
            f = sif.scan_image_features(word_image, self.numb_f, normalize_feature_matrix=True)
            features[i] = f
            words.append(w[10:-4])
            locations.append(w[:9])
            if i % 100 == 0:
                print("feature extraction, image ", i, "out of", len(word_images))
        return [[w, features[i], locations[i]] for i, w in enumerate(words)]
        # the dictionary way: but needs a lot of other changes
        # return {'words': {'word': w, 'features': features[i], 'location': locations[i]} for i, w in enumerate(words)}

    def spot_keywords(self, pages, keywords, result_file_name=''):
        file_filters = []
        for page in pages:
            file_filters.append(str(page) + '-*.png')
        word_images = sorted(fnmatch.filter(os.listdir(self.paths["resized_word_images"]), file_filters))
        validate_word_features = self.get_word_features(word_images)
        return self.spot_keywords(validate_word_features, keywords, result_file_name)

    def spot_keywords(self, validate_word_features, keywords, result_file_name=''):
        keywords_features = []
        for kwf in self.words_features:
            if kwf[0] in keywords:
                keywords_features.append(kwf)
        # we have multiple features for the same word so better sort them for better statistics
        keywords_features.sort(key=lambda e: e[0])
        self.rp.keywords = keywords

        spotted_words = []
        count_good_spots = 0
        count_bad_spots = 0
        count_missed_spots = 0

        print("Scanning %d words for %d keywords with %d word features and DTW threshold %.2f"
              % (len(validate_word_features), len(keywords), len(keywords_features), self.spot_threshold))
        for i, (true_word, wf, word_location) in enumerate(validate_word_features):
            current_word_best_dtw = None
            if i > 0 and i % 100 == 0:
                print('scanned  %d words' % i)

            for j, kwf in enumerate(keywords_features):
                spotted_word = kwf[0]
                (d, cost_matrix, acc_cost_matrix, path) = dtw.dtw(kwf[1], wf, dist=euclidean)
                if current_word_best_dtw is None or d < current_word_best_dtw[1]:
                    current_word_best_dtw = (spotted_word, d, cost_matrix, acc_cost_matrix, path)

            # check spotting condition
            spotted_word = current_word_best_dtw[0]
            d = current_word_best_dtw[1]
            spot = d < self.spot_threshold
            self.rp.add(spotted_word, true_word, spot)
            if spot:
                spotted_words.append((spotted_word, word_location, true_word, d))   # (d, cost_matrix, acc_cost_matrix, path)))
                if spotted_word.lower() == true_word.lower():
                    count_good_spots += 1
                    print("* spotted\t [%s]\t\t at [%s],\t d: %.4f" % (spotted_word, word_location, d))
                else:
                    count_bad_spots += 1
                    print("! mis-spotted\t [%s]\t as [%s]\t at [%s],\t d: %.4f" % (spotted_word, true_word, word_location, d))
            elif true_word in keywords:
                count_missed_spots += 1
                print("- not spotted\t [%s]\t\t at [%s],\t d: %.4f as [%s]" % (true_word, word_location, d, spotted_word))

        # save results
        if len(result_file_name) != 0:
            with open(result_file_name, 'w') as fr:
                writer = csv.writer(fr, lineterminator='\n')
                for word_stat in spotted_words:
                    writer.writerow(word_stat)
                writer.writerow("Stats: ")
                writer.writerow(self.rp.stats())

        print("FINAL STATS:")
        print("\t Good spots TP: %d " % count_good_spots)
        print("\t Bad spots FP: %d " % count_bad_spots)
        print("\t Missed spots FN: %d " % count_missed_spots)
        total = len(validate_word_features)
        print("\t Total scanned words: %d " % total)
        print("\t Stats: \n%s " % self.rp.stats())
        self.rp.plot()
        # TODO maybe plot outside
        return
