#!/usr/bin/python

import os
import numpy as np
import platform
import argparse

from DynTimeWrap import DynTimeWrap
from RecallPrecision import RecallPrecision

parser = argparse.ArgumentParser()
parser.add_argument('--feature_extr', default=False, type=bool)
parser.add_argument('--dtw', default=False, type=bool)
parser.add_argument('--test', default=True, type=bool)
parser.add_argument('--numb_f', default=9, type=int)
parser.add_argument('--clean_word', default=False, type=bool)
args = parser.parse_args()

# ----- paths and folders and shit ----#
work_dir = os.getcwd()
if work_dir[-14:] != "4.5_biologists" and work_dir[-3:] != "kws":
    print("get back to main directory, or cd into src/main/py/kws, sucker !")
    exit()

paths = {"resized_word_images":     os.path.join('data', 'resized_word_images'),
         "train.txt":               os.path.join('data', 'task', 'train.txt'),
         "valid.txt":               os.path.join('data', 'task', 'valid.txt'),
         "keywords.txt":            os.path.join('data', 'task', 'keywords.txt'),
         "train_features.txt":      os.path.join('data', 'train_features.txt'),
         "valid_features.txt":      os.path.join('data', 'valid_features.txt'),
         "spotting_results.txt":    os.path.join('data', 'spotting_results.txt'),
         "spotted_keywords_dtw.dump": os.path.join('data', 'spotted_keywords_dtw.dump'),
         "best_spotted_RP_plot.png": os.path.join('data', 'best_spotted_RP_plot.png'),
         "train_all_features.txt":      os.path.join('data', 'train_all_features.txt'),
         "test.txt":                os.path.join('data', 'task', 'test.txt'),
         "test_keywords.txt":       os.path.join('data', 'task', 'test_keywords.txt'),
         "test_features.txt":      os.path.join('data', 'test_features.txt'),
         "test_spotting_results.txt":    os.path.join('data', 'spotting_results.txt'),
         "test_spotted_keywords_dtw.dump": os.path.join('data', 'spotted_keywords_dtw.dump')
         }

dtw = DynTimeWrap(paths=paths, numb_f=args.numb_f)

# adapt if run from 4.5_biologists
if os.getcwd()[-14:] == "4.5_biologists":
    for k in paths:
        paths[k] = os.path.join('src', 'main', 'py', 'kws', paths[k])

# adapt for Windows
if platform.system() == "Windows":
    for k in paths:
        # paths[k] = re.sub("/", "\\\\", paths[k])
        # or
        paths[k] = os.path.normpath(paths[k])

# and create directories if these don't exist

for k in paths:
    ext = os.path.splitext(paths[k])[1]
    if not os.path.exists(paths[k]) and len(ext) == 0:
        os.makedirs(paths[k])

# ----- features extraction ----#
if args.feature_extr or not os.path.isfile(paths["train_features.txt"]):  # first time no way you have to do it
    dtw.train(train_pages=range(270, 280), save_file_name=paths['train_features.txt'])
if args.feature_extr or not os.path.isfile(paths["valid_features.txt"]):
    # create also validating word_features to speed up testing
    dtw.train(train_pages=range(300, 305), save_file_name=paths['valid_features.txt'])
if args.feature_extr or not os.path.isfile(paths["train_all_features.txt"]):
    dtw.train(train_pages=(list(range(270, 280))+list(range(300, 305))), save_file_name=paths['train_all_features.txt'])
if args.feature_extr or not os.path.isfile(paths["test_features.txt"]):
    dtw.train(train_pages=range(305, 309), save_file_name=paths['test_features.txt'])


def load_keywords(keywords_file, clean):
    keywords = []
    with open(keywords_file, "r") as pages:
        for line in pages:
            keyword = line.rstrip("\n\r")
            if clean:
                keyword = keyword.replace('-', '').replace('_cm', '').replace('_pt', '')\
                    .replace('_qo', '').replace('_', ' ')
            keywords.append(keyword)
    return keywords


# ----- dtw ----#
if args.dtw:
    dtw.train_word_features(paths['train_features.txt'])

    print("time for dynamic time warping...")

    # keywords = ['Alexandria', 'Captain', 'Colonel']
    # keywords = ['Alexandria', 'Captain', 'Colonel', 'Lieutenant', 'Major', 'Letters', 'October']
    keywords = load_keywords(paths["keywords.txt"], clean=args.clean_word)

    # used saved valid set: faster
    valid_word_features = dtw.load_word_features(paths['valid_features.txt'])
    spotted = dtw.spot_keywords(valid_word_features, keywords)
    # or do it now
    # spotted = dtw.spot_keywords_in_pages([300], keywords)

    dtw.dump_spotted_keywords_dtw(paths['spotted_keywords_dtw.dump'])   # could be used for later plots
    dtw.report_spotted_keywords(paths['spotting_results.txt'])

    d_threshold = 6.0  # so far best found
    rp = RecallPrecision(keywords)
    for d_threshold in np.arange(4.0, 9.0, 0.2):
        best_spotted = dtw.best_spotted_keywords(d_threshold=d_threshold)

        print('Found %d best spotted words with d_threshold %.4f :' % (len(best_spotted), d_threshold))
        #print('location,\t keyword,\t true_word,\t distance' % d_threshold)
        for (loc, d, keyword, true_word) in best_spotted:
            rp.add(keyword, true_word, True)  # they are all calls
            #if keyword.lower() != true_word.lower():   # interested only in mis-matches
            #    print('%s,\t%s,\t%s,\t%.4f' % (loc, keyword, true_word, d))
        print("Stats: \n\t %s " % rp.stats())

        rp.add_plot_point()

    rp.plot(paths['best_spotted_RP_plot.png'])

# ----- test set ----#
if args.test:
    dtw.train_word_features(paths['train_all_features.txt'])

    keywords = load_keywords(paths["test_keywords.txt"], clean=args.clean_word)

    test_word_features = dtw.load_word_features(paths['test_features.txt'])
    spotted = dtw.spot_keywords(test_word_features, keywords)

    dtw.dump_spotted_keywords_dtw(paths['test_spotted_keywords_dtw.dump'])   # could be used for later plots
    dtw.report_spotted_keywords(paths['test_spotting_results.txt'])
