#!/usr/bin/python

"""
load all images from data/images,
binerize every image and save it in a new folder

Its hard to find a threshold that works good.
Maybe i'll try a different way of doing it
"""
import scipy.misc
import numpy as np
import os
import matplotlib.pyplot as plt

#path of the folder containing the original images
input_path = 'src/main/py/kws/data/images'
#path of the folder in which the processed images should be saved
output_path = 'src/main/py/kws/data/binarized_images'

# create directory if not existant
if not os.path.exists(output_path):
    os.makedirs(output_path)

#get the name of every image file
list_of_images = os.listdir(input_path)
#process one image after the other
for file in list_of_images:
    # load image
    image = plt.imread(input_path + '/' + file)
    # set threshold for binarization (you can try different ones)
    threshold = image.max()*0.76
    #binarization (every pixel aboce the threshold will be set to one and all others to 0
    image = np.where(image > threshold,1, 0)
    #save the processed file
    scipy.misc.imsave(output_path + '/' + file, image)

