#!/usr/bin/python

import scipy.misc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

def resize_images() :
    input_path = r"C:\Users\svenw\Desktop\GitHub\4.5_biologists\src\main\py\kws\data\word_images"
    output_path = r"C:\Users\svenw\Desktop\GitHub\4.5_biologists\src\main\py\kws\data\resized_word_images"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    list_of_words = os.listdir(input_path)


    for file in list_of_words:
        img = plt.imread(input_path + '/' + file) #the loaded image has the shape: (height, widht, 4)

        #these steps remove the 4 channels in the 3. dimension
        frame = img[:, :, 3]
        img = img[:, :, 0]
        img = np.where((frame[:, :] == 0), 1, img[:, :])

        resized_img = cv2.resize(img, (100, 100)) #the reshaped image has the shape: (100, 100)

        scipy.misc.imsave(output_path + '/' + file, resized_img)



def median_wh(list_of_wordimages):
    """Returns median width and height of images present in a list of filenames (e.g. created
    using os.listdir). Must be run in the respective directory where the files are present."""
    word_widths = list()
    word_heights = list()
    for word in list_of_wordimages:
        image = plt.imread(word)
        word_widths.append(image.shape[1])
        word_heights.append(image.shape[0])
    median_word_width = int(np.median(word_widths))  # 207 is the median width of the cut words
    median_word_height = int(np.median(word_heights))  # 207 is the median width of the cut words
    return median_word_width, median_word_height

# os.chdir(paths["wordimages_input"])
#list_of_wordimages = os.listdir(paths["wordimages_input"])
    #list_of_wordimages = os.listdir("C:\\Data\\word_images")
# median_wh(list_of_wordimages)
