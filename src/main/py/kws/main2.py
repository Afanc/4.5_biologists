#!/usr/bin/python

import os, numpy as np
import argparse
import read_transcription as rt
import read_SVG as svgread
import crop_svg_outline as svgcrop
import binarization as binary
import feature_extraction as features
import scan_image_features as scan
from matplotlib import pyplot as plt
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--preprocessing', default=True, type=bool)
parser.add_argument('--id_linking', default=True, type=bool)
args = parser.parse_args()

#----- paths and folders and shit ----#
work_dir = os.getcwd()
if(work_dir[-14:] != "4.5_biologists" and work_dir[-3:] != "kws"):
    print("get back to main directory, or cd into src/main/py/kws, sucker !")
    exit()

paths= {}

paths["images_input"] = 'data/images'
paths["images_output"] = 'data/binarized_images'
paths["wordimages_input"] = "data/word_images"
paths["wordimages_output"] = "data/resized_word_images"

if(os.getcwd()[-14:] == "4.5_biologists"):
    for k in paths :
        paths[k] = "src/main/py/kws/" + paths[k]

#exit()

for p in paths :
    if not os.path.exists(p):
        os.makedirs(p)

transcription_path = os.path.normpath(".\\data\\ground-truth\\")
svg_path = os.path.normpath(".\\data\\ground-truth\\locations")
image_path = os.path.normpath(".\\data\\images")

transcript_file = os.path.join(transcription_path, "transcription.txt")

list_of_images = os.listdir(image_path)
list_of_svg = os.listdir(svg_path)


#----- ID linking----#
if args.id_linking :

    word_dict = rt.read_transcription(file_name = transcript_file, output = "word_dict")
    ID_dict =  rt.read_transcription(file_name = transcript_file, output = "ID_dict")


#----- pre-processing ----#
if args.preprocessing :

    for page_no in range(len(list_of_images)):
        svg = os.path.normpath(os.path.join(svg_path, list_of_svg[page_no]))
        image = os.path.normpath(os.path.join(image_path, list_of_images[page_no]))
        coord_list = svgread.extract_SVG_masks(svg)
        svgcrop.crop_svg_outline(image, ID_dict = ID_dict, svg_coordinates = coord_list)


    list_of_wordimages = os.listdir(paths["wordimages_input"])
    word_lengths = [len(word) for word in word_dict]
    median_word_length = int(np.median(word_lengths))

    for file in list_of_wordimages:
        file_in = os.path.normpath(os.path.join(paths["wordimages_input"], file))
        img = plt.imread(file_in) #the loaded image has the shape: (height, widht, 4)

     #these steps remove the 4 channels in the 3. dimension
        frame = img[:, :, 3]
        img = img[:, :, 0]
        img = np.where((frame[:, :] == 0), 1, img[:, :])
#        img = plt.imread("270-25-05_Clock.png")
        img_PIL = Image.fromarray(img)
        resized_img = img_PIL.resize(size = (100, 200)) #the reshaped image has the shape: (100, 100)
        resized_img_array = np.array(resized_img)
        file_out = os.path.normpath(os.path.join(paths["wordimages_output"], file))
#        resized_img.save(file_out)
#        scipy.misc.imsave(file_out, resized_img)
        resized_img_bin = binary.binarize_image(resized_img_array, block_size = 5)
        binary.save_image_png(file_out, resized_img_bin)

