#!/usr/bin/python

import os
# import csv
import pickle
from wcmatch import fnmatch
import numpy as np
# import pandas as pd
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
    def __init__(self, paths, numb_f=9, f_width=212, spot_threshold=7.):   # TODO find the better threshold
        self.paths = paths
        self.numb_f = numb_f
        self.f_width = f_width
        self.words_features = []
        self.learned_words = set()
        self.spot_threshold = spot_threshold
        self.keywords = []
        self.spotted_keywords_dtw = {}

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

    def spot_keywords_in_pages(self, pages, keywords):
        file_filters = []
        for page in pages:
            file_filters.append(str(page) + '-*.png')
        word_images = sorted(fnmatch.filter(os.listdir(self.paths["resized_word_images"]), file_filters))
        validate_word_features = self.get_word_features(word_images)
        return self.spot_keywords(validate_word_features, keywords)

    def spot_keywords(self, validate_word_features, keywords):
        keywords_features = []
        for kwf in self.words_features:
            if kwf[0] in keywords:
                keywords_features.append(kwf)
        # we have multiple features for the same word so better sort them for better statistics
        keywords_features.sort(key=lambda e: e[0])
        self.keywords = keywords

        print("Scanning %d words for %d keywords with %d training words"
              % (len(validate_word_features), len(keywords), len(keywords_features)))

        # dtws_df = pd.DataFrame(columns=keywords + ['__true_word', '__location'])   # panda way...

        for i, (keyword, keyword_features, kw_location) in enumerate(keywords_features):
            if i > 0 and i % 10 == 0:
                print('scanned %d training words' % i)

            keyword_dtw = self.spotted_keywords_dtw[keyword] if keyword in self.spotted_keywords_dtw else []
            for j, (true_word, word_features, word_location) in enumerate(validate_word_features):
                (d, cost_matrix, acc_cost_matrix, path) = dtw.dtw(keyword_features, word_features, dist=euclidean)
                if len(keyword_dtw) == j:
                    # first time with this keyword
                    keyword_dtw.append([word_location, d, true_word])
                elif d < keyword_dtw[j][1]:
                    # found better match
                    assert keyword_dtw[j][0] == word_location, "locations doesn't match: %s %s" % (keyword_dtw[j][0], word_location)
                    assert keyword_dtw[j][2] == true_word, "words doesn't match: %s %s" % (keyword_dtw[j][2], true_word)
                    keyword_dtw[j] = [word_location, d, true_word]
                # dtws_df.append()

            self.spotted_keywords_dtw[keyword] = keyword_dtw

        return self.spotted_keywords_dtw

    def save_spotted_keywords(self, file_name):
        """ Save all spotted keywords in the required format:
            - one line per keyword: locations sorted by dtw distance
                <keyword>,<location_1>,<dist_1>,<location_2><dist_2>...
        :param file_name: where to save
        """
        if len(file_name) != 0:
            with open(file_name, 'w') as fr:
                for keyword, keyword_dtw in self.spotted_keywords_dtw.items():
                    # sort by dtw distance <d>
                    keyword_dtw.sort(key=lambda w: w[1])
                    dtws = [(w[0] + ',' + str(round(w[1], 4))) for w in keyword_dtw]
                    fr.write(keyword + ',' + ','.join(dtws) + '\n')

    def best_spotted_keywords(self, d_threshold):
        n = 0
        for keyword, keyword_dtw in self.spotted_keywords_dtw.items():
            # sort by word location
            keyword_dtw.sort(key=lambda w: w[0])
            n = max(n, len(keyword_dtw))

        # find best for location
        best_spotted = []    # something like this: np.full((n), ['', float("inf"), ''])
        for keyword, keyword_dtw in self.spotted_keywords_dtw.items():
            # find best
            for i, (loc, d, true_word) in enumerate(keyword_dtw):
                if len(best_spotted) == i:
                    # first time
                    best_spotted.append([loc, d, keyword, true_word])
                elif d < best_spotted[i][1]:
                    # found better
                    assert best_spotted[i][0] == loc, " locations doesn't match: %s %s" % (best_spotted[i][0], loc)
                    best_spotted[i] = [loc, d, keyword, true_word]

        # threshold by distance
        return [bs for bs in best_spotted if bs[1] < d_threshold]

    def recall_precision_stats(self):
        rp = RecallPrecision(self.keywords)
        rp.keywords = self.keywords
        th_min = float("inf")
        th_max = 0
        for keyword, keyword_dtw in self.spotted_keywords_dtw.items():
            dtw_d = [w[1] for w in keyword_dtw]
            w_min = min(dtw_d)
            th_min = min(th_min, w_min)
            w_max = max(dtw_d)
            th_max = max(th_max, w_max)

        for th in np.linspace(th_min, th_max, 100):   # or range(th_min, th_max, 0.1)
            for keyword, keyword_dtw in self.spotted_keywords_dtw.items():
                dtws_spotted = [w for w in keyword_dtw if w[1] < th]
                for w in dtws_spotted:
                    rp.add(keyword, w[2], True)
                rp.add_plot_point()

            rp.plot()

        print("FINAL STATS:")
        print("\t %s " % rp.stats())
        rp.plot()
        # TODO maybe plot outside
        return
