#!/usr/bin/python

import os


class DynTimeWrap:
    """Dynamic Time Warping between two feature vector sequences"""
    def __init__(self, word='Letters', word_dict=os.path.join('kws', 'data', 'word_positionID_dict.txt')):
        self.word = word
        self.word_dict = word_dict

