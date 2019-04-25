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
                 "validate.txt": os.path.join('data', 'task', 'validate.txt'),
                 "keywords.txt": os.path.join('data', 'task', 'keywords.txt'),
                 "train_features.txt": os.path.join('data', 'train_features.txt')
                 }
        self.dtw = DynTimeWrap(self.paths)

    def test_train(self):
        train_pages = []
        with open(self.paths["train.txt"], "r") as train:
            for line in train:
                page = line.rstrip("\n")
                train_pages.append(page)
        features = self.dtw.train(train_pages, save_file_name=self.paths['train_features.txt'])
        assert len(features) > 0

    def test_load_word_features(self):
        features = self.dtw.load_word_features(self.paths['train_features.txt'])
        assert len(features) > 0
