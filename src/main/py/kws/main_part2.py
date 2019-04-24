#!/usr/bin/python

import os
import numpy as np
import platform
import argparse
import feature_extraction as features
import scan_image_features as scan
from matplotlib import pyplot as plt
import scan_image_features as sif

parser = argparse.ArgumentParser()
parser.add_argument('--feature_extr', default=True, type=bool)
parser.add_argument('--dtw', default=True, type=bool)
parser.add_argument('--numb_f', default=4, type=int)
parser.add_argument('--width', default=207, type=int)
args = parser.parse_args()

# ----- paths and folders and shit ----#
work_dir = os.getcwd()
if (work_dir[-14:] != "4.5_biologists" and work_dir[-3:] != "kws"):
    print("get back to main directory, or cd into src/main/py/kws, sucker !")
    exit()

paths = {}

# directories
paths["wordimages_input"] = os.path.join('data', 'resized_word_images')
paths["svg"] = os.path.join('data', 'ground-truth', 'locations')

# files
paths["transcription.txt"] = os.path.join('data', 'ground-truth', 'transcription.txt')

# adapt if run from 4.5_biologists
if (os.getcwd()[-14:] == "4.5_biologists"):
    for k in paths:
        paths[k] = os.path.join('src', 'main', 'py', 'kws', paths[k])

# adapt for Windows
if (platform.system() == "Windows"):
    for k in paths:
        # paths[k] = re.sub("/", "\\\\", paths[k])
        # or
        paths[k] = os.path.normpath(paths[k])

# and create directories if these don't exist
for k in paths:
    if (not os.path.exists(paths[k]) and paths[k][:-4] != ".txt"):
        os.makedirs(paths[k])


list_of_wordimages = sorted(os.listdir(paths["wordimages_input"]))
list_of_svg = sorted(os.listdir(paths["svg"]))

# ----- features extraction ----#
if args.feature_extr:
    number_of_features = args.numb_f
    width = args.width

    features = np.zeros(shape=(len(list_of_wordimages), number_of_features, width))
    print(features.shape)

    for i,w in enumerate(list_of_wordimages) : 
        wordimage = os.path.join(paths["wordimages_input"], w)
        f = sif.scan_image_features(wordimage, number_of_features, normalize_feature_matrix=True)
        features[i] = f

        print(features[i])
        break

#TODO
#extract features to a csv

# ----- pre-processing ----#
if args.dtw:
    print(features[0])

