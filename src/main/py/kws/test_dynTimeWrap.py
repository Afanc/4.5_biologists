#!/usr/bin/python

from unittest import TestCase
from DynTimeWrap import DynTimeWrap

import os

class TestDynTimeWrap(TestCase):
    def setUp(self):
        self.paths = {"images": os.path.join('data', 'images'),
                        "binarized_images": os.path.join('data', 'binarized_images'),
                        "word_images": os.path.join('data', 'word_images'),
                        "resized_word_images": os.path.join('data', 'resized_word_images'),
                        "svg": os.path.join('data', 'ground-truth', 'locations'),
                        "transcription.txt": os.path.join('data', 'ground-truth', 'transcription.txt'),
                        "train.txt": os.path.join('data', 'task', 'train.txt'),
                        "valid.txt": os.path.join('data', 'task', 'valid.txt'),
                        "keywords.txt": os.path.join('data', 'task', 'keywords.txt'),
                        "train_features.txt": os.path.join('data', 'train_features.txt'),
                        "spotting_results.txt": os.path.join('data', 'csv_results.txt')
                    }
        self.dtw = DynTimeWrap(self.paths)

    def test_train(self):
        train_pages = []
        with open(self.paths["train.txt"], "r") as pages:
            for line in pages:
                page = line.rstrip("\n\r")
                train_pages.append(page)
        features = self.dtw.train(train_pages, save_file_name=self.paths['train_features.txt'])
        assert len(features) > 0

    def test_save_load_word_features(self):
        features = self.dtw.train(['270'])  # , save_file_name=self.paths['train_features.txt'])
        print('trained features: ' + str(len(features)) + ' to save')
        self.dtw.save_word_features(features, self.paths['train_features.txt'])
        loaded_features = self.dtw.load_word_features(self.paths['train_features.txt'])
        print('loaded saved features: ' + str(len(loaded_features)))
        assert len(loaded_features) == len(features)

    def test_spot_keywords(self):
        # first train
        self.dtw.load_word_features(self.paths['train_features.txt'])
        # self.test_train()  # train in memory
        # then spot
        spot_pages = []
        with open(self.paths["valid.txt"], "r") as pages:
            for line in pages:
                page = line.rstrip("\n\r")
                spot_pages.append(page)
        spotted = self.dtw.spot_keywords(spot_pages, ['Letters', 'October'], self.paths['spotting_results.txt'])
        print(spotted)
