#!/usr/bin/python

import os
import platform
import argparse

from DynTimeWrap import DynTimeWrap
from RecallPrecision import RecallPrecision

parser = argparse.ArgumentParser()
parser.add_argument('--feature_extr', default=False, type=bool)
parser.add_argument('--dtw', default=True, type=bool)
parser.add_argument('--numb_f', default=9, type=int)
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
         "spotting_results.txt":    os.path.join('data', 'spotting_results.txt')
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
    if not os.path.exists(paths[k]) and paths[k][-4:] != ".txt":
        os.makedirs(paths[k])

# ----- features extraction ----#
if args.feature_extr or not os.path.isfile(paths["train_features.txt"]):  # first time no way you have to do it
    # create also validating word_features to speed up testing
    dtw.train(train_pages=range(300, 305), save_file_name=paths['valid_features.txt'])
    dtw.train(train_pages=range(270, 280), save_file_name=paths['train_features.txt'])
else:
    dtw.train_word_features(paths['train_features.txt'])


# ----- dtw ----#
if args.dtw:

    print("time for dynamic time warping...")

    def load_keywords(clean):
        keywords = []
        with open(paths["keywords.txt"], "r") as pages:
            for line in pages:
                keyword = line.rstrip("\n\r")
                if clean:
                    keyword = keyword.replace('-', '').replace('_cm', '').replace('_pt', '')\
                        .replace('_qo', '').replace('_', ' ')
                keywords.append(keyword)
        return keywords

    # keywords = ['Alexandria', 'Captain', 'Colonel']
    # keywords = ['Alexandria', 'Captain', 'Colonel', 'Lieutenant', 'Major', 'Letters', 'October']
    keywords = load_keywords(clean=True)

    # used saved valid set: faster
    valid_word_features = dtw.load_word_features(paths['valid_features.txt'])
    spotted = dtw.spot_keywords(valid_word_features, keywords)
    # or do it now
    # spotted = dtw.spot_keywords_in_pages([300], keywords)

    dtw.save_spotted_keywords(paths['spotting_results.txt'])

    d_threshold = 6.0
    best_spotted = dtw.best_spotted_keywords(d_threshold=d_threshold)

    rp = RecallPrecision(keywords)
    print('Best spotted with d_threshold %.4f :' % d_threshold)
    print('location,\t keyword,\t true_word,\t distance' % d_threshold)
    for (loc, d, keyword, true_word) in best_spotted:
        print('%s,\t%s,\t%s,\t%.4f' % (loc, keyword, true_word, d))
        rp.add(keyword, true_word, True)  # they are all calls

    print("Stats: \n\t %s " % rp.stats())


