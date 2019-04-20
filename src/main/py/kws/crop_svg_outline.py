#!/usr/bin/python

import os
from PIL import Image, ImageDraw
import numpy as np
from skimage import color
from skimage import io

# https://pypi.org/project/svgpathtools/

def crop_svg_outline(image_file, ID_dict, svg_coordinates):
    """Accepts image, ID-dictionary (key: position in format page-line-word; values: word), and svg-masks.
    Outlines all words for which an svg-mask is available, crops the outlined word and saves a file with name in the
    format "positional ID_literal word.png". """
    output_path_polygon = ".\\data\\word_images"
    if not os.path.exists(output_path_polygon):
        os.makedirs(output_path_polygon)

    img = Image.open(image_file).convert("RGBA")

    for entry in svg_coordinates:
        ID = entry[0]
        outline = entry[1]
        outline_polygon = entry[2]
        word = ID_dict[ID][2]  # the literal word at this position

        # define polygon mask
        imArray = np.asarray(img)
        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(outline_polygon, outline=1, fill=1)
        mask = np.array(maskIm)

        # assemble new image (uint8: 0-255)
        newImArray = np.empty(imArray.shape, dtype='uint8')

        # colors (three first columns, RGB)
        newImArray[:, :, :3] = imArray[:, :, :3]

        # transparency (mask)
        newImArray[:, :, 3] = mask * 255

        # define bounding box for cropping (UL upper left and LR lower right)
        ULx = np.amin(outline, axis = 0)[0]
        ULy = np.amax(outline, axis = 0)[1]
        LRx = np.amax(outline, axis = 0)[0]
        LRy = np.amin(outline, axis = 0)[1]

        img_crop_polygon = newImArray[LRy:ULy, ULx:LRx]

        # reconvert to Image format for saving
        # newIm = Image.fromarray(img_crop_polygon, "LA")  # Leaving RGBA seems unintuitive, but the main.py then removes mask and extra channels (could not incorporate this into the f...ing !!! function, maybe if we have a lot of time left...)
        newIm = Image.fromarray(img_crop_polygon, "RGBA")
        #print(type(newIm))
        #newIm = color.rgb2gray(io.imread(newIm))
        file_name = ID +"_" + word + ".png"
        save_path = os.path.join(output_path_polygon, file_name)
        newIm.save(save_path)
    return

# os.getcwd()
# os.chdir("C:\\Bern\\Github\\4.5_biologists-master\\src\\main\\py\\kws")

# crop_svg_outline("270.jpg", ID_dict = ID_dict, svg_coordinates = coord_list)
