#!/usr/bin/python

from unittest import TestCase

import numpy as np
import scan_image_features as sif
from PIL import Image

class TestScan_image_features(TestCase):
    def test_scan_image_features(self):
        img = Image.open("270-25-05_Clock.png")

        features_Clock = sif.scan_image_features("270-25-05_Clock.png", normalize_feature_matrix=False)
        print(np.array(img).shape)
        print(features_Clock.shape)
        # same as:
        features_Clock2 = sif.scan_image_features("270-25-05_Clock.png")
        features_Clock_norm = sif.scan_image_features("270-25-05_Clock.png", normalize_feature_matrix=True)
        # self.fail()
