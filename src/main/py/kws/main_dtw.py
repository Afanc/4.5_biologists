#!/usr/bin/python

import os
import platform
import argparse

from DynTimeWrap import DynTimeWrap

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
    dtw.train(train_pages=range(270, 280), save_file_name=paths['train_features.txt'])
else:
    dtw.load_word_features(paths['train_features.txt'])


def load_keywords(self):
    keywords = []
    with open(self.paths["keywords.txt"], "r") as pages:
        for line in pages:
            keyword = line.rstrip("\n\r").replace('-', '').replace('_cm', '').replace('_pt', '') \
                .replace('_qo', '').replace('_', ' ')
            keywords.append(keyword)
    return keywords


# keywords = ['Alexandria', 'Letters', 'October']
keywords = ['Alexandria', 'Captain', 'Colonel', 'Lieutenant', 'Major', 'Letters', 'October']
# keywords = load_keywords()

spot_pages = range(300, 305)


# ----- dtw ----#
if args.dtw:

    print("warping : wwwwwooooooooooo......")

    spotted = dtw.spot_keywords(spot_pages, keywords, paths['spotting_results.txt'])
