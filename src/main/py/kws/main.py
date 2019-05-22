#!/usr/bin/python

import os, numpy as np
import platform
import argparse
import read_transcription as rt
import read_SVG as svgread
import crop_svg_outline as svgcrop
import resize_images as resize
import binarization as binary
# import feature_extraction as features
# import scan_image_features as scan
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessing', default=True, type=bool)
parser.add_argument('--id_linking', default=True, type=bool)
parser.add_argument('--word_width', default=212, type=int)
parser.add_argument('--word_height', default=94, type=int)
args = parser.parse_args()

# ----- paths and folders ----#
work_dir = os.getcwd()
if (work_dir[-14:] != "4.5_biologists" and work_dir[-3:] != "kws"):
    print("get back to main directory, or cd into src/main/py/kws, sucker !")
    exit()

paths = {}

# directories
paths["images"] = os.path.join('data', 'images')
paths["binarized_images"] = os.path.join('data', 'binarized_images')
paths["word_images"] = os.path.join('data', 'word_images')
paths["resized_word_images"] = os.path.join('data', 'resized_word_images')
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


list_of_images = sorted(os.listdir(paths["images"]))
list_of_svg = sorted(os.listdir(paths["svg"]))


# ----- pre-processing ----#
if args.preprocessing:
    if not os.listdir(paths["binarized_images"]) or not os.listdir(paths["word_images"]):
        # ----- ID linking----#
        ID_dict = rt.read_transcription(file_name=paths["transcription.txt"], output="ID_dict")

        # --- processing pages (binarization and cropping out words) --- #
        i = 0
        for page_no, page in enumerate(list_of_images):
            print("processing page ", i+1, " out of ", len(list_of_images))

            image = plt.imread(os.path.join(paths["images"], page))
            svg = os.path.join(paths["svg"], list_of_svg[page_no])
            coord_list = svgread.extract_SVG_masks(svg)

            img_name = page[:-4] + ".png"
            image_out = os.path.join(paths["binarized_images"], img_name)

            image_bin = binary.binarize_image(image, block_size=101)  # binarize image using local thresholding
            binary.save_image_png(image_out, image_bin)

            svg_in = os.path.join(paths["binarized_images"], img_name)
            svgcrop.crop_svg_outline(svg_in, ID_dict=ID_dict, svg_coordinates=coord_list, output_path=paths["word_images"])  # crop individual words by polygon outline

            i += 1


    # --- get median word width and height (for resizing) --- #
    base = os.getcwd()
    list_of_wordimages = sorted(os.listdir(paths["word_images"]))
    list_of_wordimages = [word for word in list_of_wordimages if not word.startswith('.')]  # ignore special files starting with '.'
    os.chdir(paths["word_images"])
    median_word_width, median_word_height = resize.median_wh(list_of_wordimages)  #    word_lengths = [len(word) for word in word_dict]
    os.chdir(base)


    # --- processing individual word images (resizing) --- #
    if not os.listdir(paths["resized_word_images"]):
        i = 0
        for file in list_of_wordimages:
            if i%100 == 0:
                print("processing word-image ", i+1, " out of ", len(list_of_wordimages))
            file_in = os.path.join(paths["word_images"], file)
            resize.resize_image(file_in, height_new=args.word_height, width_new=args.word_width, output_path=paths["resized_word_images"])

            i += 1

    print("Binary images of individual words extracted and rescaled to", args.word_width, "x", args.word_height, "pixel (width x height).")
    print("Medium is", median_word_width, "x", median_word_height, "pixel (width x height).")
