#!/usr/bin/python

import os
import numpy as np
import platform
import argparse
import feature_extraction as features
import scan_image_features as scan
from matplotlib import pyplot as plt
import scan_image_features as sif
import read_transcription as rt
import dyn_time_warp as dtw
from itertools import combinations
import csv
import random

parser = argparse.ArgumentParser()
parser.add_argument('--feature_extr', default=True, type=bool)
parser.add_argument('--dtw', default=True, type=bool)
parser.add_argument('--numb_f', default=9, type=int)
parser.add_argument('--width', default=212, type=int)
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
paths["features.txt"] = os.path.join('data', 'features.txt')
paths["csv_results.txt"] = os.path.join('data', 'csv_results.txt')
paths["word_positionID_dict.txt"] = os.path.join('data', 'word_positionID_dict.txt')
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
    if (not os.path.exists(paths[k]) and paths[k][-4:] != ".txt"):
        os.makedirs(paths[k])

list_of_wordimages = sorted(os.listdir(paths["wordimages_input"]))
list_of_svg = sorted(os.listdir(paths["svg"]))

# ----- features extraction ----#
words_and_features = []

if args.feature_extr:
    number_of_features = args.numb_f
    width = args.width

    features = np.zeros(shape=(len(list_of_wordimages), number_of_features, width))
    words = []
    #print(features.shape)

    for i, w in enumerate(list_of_wordimages):
        if w.startswith('.'):
            continue
        wordimage = os.path.join(paths["wordimages_input"], w)
        f = sif.scan_image_features(wordimage, number_of_features, normalize_feature_matrix=True)
        features[i] = f

        words.append(w[10:-4])

        #testing
        if(i>10) : 
            pass 

        if(i%100 == 0) :
            print("feature extraction, image ", i, "out of", len(list_of_wordimages))

    words_and_features = [[w, features[i]] for i,w in enumerate(words)]

    if os.path.isfile(paths["features.txt"]):
        with open(paths["features.txt"], 'w') as f:
            writer = csv.writer(f , lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
            for row in words_and_features:
                writer.writerow(row)


# --- get a random list of word for testing -- #
def get_test_words(num):
    if not os.path.isfile(paths["word_positionID_dict.txt"]):
        words = rt.read_transcription(file_name=paths["transcription.txt"], output="word_dict")
        words = words.keys()
    else:
        with open(paths["word_positionID_dict.txt"], 'r') as f:
            reader = csv.reader(f, delimiter=',')
            word_dict = list(reader)
            words = [word[0] for word in word_dict]
    rand_words = random.sample(words, num)
    return rand_words


test_words = get_test_words(3)

# ----- dtw ----#
if args.dtw:

    print("warping : wwwwwooooooooooo......")

    #TODO
    #this part is shitty, if we want to extract those features to a csv........... since multiple rows and shit
    #ach... anyone has any idea ?
    #and then if it's fast, we can even merge args.feature_extr and args.dtw 
    #read from csv
    if len(words_and_features) == 0:
        with open(paths["features.txt"], 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                words_and_features.append(row)

    #print(words_and_features[0])

    #so this might depend on how we write the whole thing, but you'll get the idea. example :
    #-----------------------this should be replaced with whole file version
    my_features = features[0:10]
    words = get_test_words(33)     # ['s_2', 'Letters', 'whatev', 'whatev', 'whatev', 'whatev', 'whatev', 'whatev', 'whatev', 'whatev']
    #------------------------

    words_and_features = [[words[i], f] for i,f in enumerate(my_features)]
    #print(words_and_features[0])
    #------------------

    #print(words_and_features[0])

    #NEEDS : list such as [['word', np.array(4,207)], ['word', np.array(4,207)], ...]

    dtw_res = [((x[0],y[0], dtw.dyn_time_warp(x[1],y[1]))) for x,y in combinations(words_and_features, 2)]

    print("exporting to file")

    with open(paths["csv_results.txt"], 'w') as f:
        writer = csv.writer(f , lineterminator='\n')
        i = 0
        for row in dtw_res:
            i+=1
            if(i%1000 == 0) :
                print("row ", i, "out of", len(dtw_res))

            writer.writerow(row)

