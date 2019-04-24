#!/usr/bin/python

import scipy.misc
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def resize_image(file, height_new, width_new, output_path = None):
    """ Loads an image, resizes it to the (new) given dimensions and saves it in the output_path. """

    if output_path is None:
        output_path = os.path.join(".", "data", "resized_word_images")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = os.path.normpath(file).split(os.path.sep)[-1]  # split path to its components and get the last of them (i.e. the file name)
    file_out = os.path.join(output_path, file_name)  # create outputpath + file_name

    img_PIL = Image.open(file)
    resized_img = img_PIL.resize(size=(width_new, height_new))  # reshaping the image
    resized_img.save(file_out)

    return



def median_wh(list_of_wordimages):
    """Returns median width and height of images present in a list of filenames (e.g. created
    using os.listdir). Must be run in the respective directory where the files are present."""
    word_widths = list()
    word_heights = list()
    for word in list_of_wordimages:
        image = plt.imread(word)
        word_widths.append(image.shape[1])
        word_heights.append(image.shape[0])
    median_word_width = int(np.median(word_widths))  # 212 is the median width of the cut words
    median_word_height = int(np.median(word_heights))  # 94 is the median height of the cut words
#    print(median_word_width, median_word_height)
    return median_word_width, median_word_height

# os.chdir(paths["wordimages_input"])
#list_of_wordimages = os.listdir(paths["wordimages_input"])
    #list_of_wordimages = os.listdir("C:\\Data\\word_images")
# median_wh(list_of_wordimages)
