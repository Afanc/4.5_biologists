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
from skimage.filters import threshold_local
from PIL import Image


def binarize_image(image, block_size = 101):
    # block_size = 101
    # use threshold adapted to local neighborhood, every pixel above the threshold will be set to 1, all others are set to 0
    if block_size % 2 == 0:  # blocksize is even
        block_size += 1  # increase block_size by 1 (because it must be odd)
    binary_image = np.where(image > threshold_local(image, block_size, offset=10), 255, 0)  # instead of ..., 1, 0)
    binary_image = binary_image.astype(np.uint8)  
    return binary_image


def save_image_png(file_name, array):
    file_name = file_name[:-4] + ".png"
    image_PIL = Image.fromarray(array)
    image_PIL.save(file_name)
    return


if __name__ == "__main__":

    #path of the folder containing the original images
    #path of the folder in which the processed images should be saved
    if(os.getcwd()[-14:] == "4.5_biologists"):
        input_path = 'src/main/py/kws/data/images'
        output_path = 'src/main/py/kws/data/binarized_images'
    elif(os.getcwd()[-3:] == "kws") :
        input_path = 'data/images'
        output_path = 'data/binarized_images'
    else :
        print("get back to main directory, or cd into src/main/py/kws, sucker !")
        exit()

    # create directory if not existant
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #get the name of every image file
    list_of_images = os.listdir(input_path)
    #process one image after the other
    for file in list_of_images:
        # load image
        image = plt.imread(input_path + '/' + file)

        # # set threshold for binarization (you can try different ones)
        # threshold = image.max()*0.76
        # #binarization (every pixel aboce the threshold will be set to one and all others to 0
        # image = np.where(image > threshold,1, 0)

        # specify window (101x101) for which the threshold will be adapted to
        block_size = 101
        # use threshold adapted to local neighborhood, every pixel above the threshold will be set to 1, all others are set to 0
#        binary_image = np.where(image > threshold_local(image, block_size, offset=10), 1, 0)  # original code --> not exactly the same as the binarize_image function (which currently assigns 255 and 0 and then converts dtype to uint8)
        binary_image = binarize_image(image, block_size)  # using the code from previous line (packed into function, so it can be imported elsewhere)

        #save the processed file
#        scipy.misc.imsave(output_path + '/' + file, binary_image)
        # consider using imageio.imwrite instead (scipy.misc.imsave is deprecated)!
#        imageio.imwrite(output_path + '/' + file, binary_image)
        # Both cause problems so far, images are not black and white when reloaded but have grayscales in addition!
                # The problem was saving as .jpg ! >:-( png or tif both work (png-files are much smaller)
        output_path = os.path.normpath(output_path)
        path_and_file = os.path.join(output_path, file)
        save_image_png(path_and_file, binary_image)
