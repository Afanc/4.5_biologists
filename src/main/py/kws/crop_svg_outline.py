#!/usr/bin/python

import os
import PIL
from PIL import Image, ImageDraw
import numpy as np
from skimage import color
from skimage import io

# https://pypi.org/project/svgpathtools/

def crop_svg_outline(image_file, ID_dict, svg_coordinates, output_path=None):
    """ Accepts (binarized) image (of a scanned page), ID-dictionary (key: position in format page-line-word; values: word), and svg-masks.
    Outlines all words for which an svg-mask is available, crops the outlined word, first along svg-polygon, then along bounding box of polygon.
    Removes mask and saves a binary image with name in the format "positional ID_literal word.png" at the given outputpath. """
    if output_path is None:
        output_path = os.path.join(".", "data", "word_images")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img = Image.open(image_file).convert("RGBA")

    for entry in svg_coordinates:
        ID = entry[0]
        outline = entry[1]
        outline_polygon = entry[2]
        if ID in ID_dict:
            word = ID_dict[ID]  # [2]  # the literal word at this position
            file_name = ID + "_" + word + ".png"
            save_path = os.path.join(output_path, file_name)
            if not os.path.isfile(save_path):

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

                frame = newImArray[:, :, 3]  # the alpha channel of the image, containing the mask with the polygon delimiter
                new_img = newImArray[:, :, 0]  # the image (in principle, the first of the three RGB-channels - which are all equivalent in case of a grayscale or binary image as input)
                newImArray = np.where((frame[:, :] == 0), 255, new_img[:, :])  # everything outside the mask is replaced by white pixels (value 255)

                # define bounding box for cropping (UL upper left and LR lower right)
                ULx = np.amin(outline, axis = 0)[0]
                ULy = np.amax(outline, axis = 0)[1]
                LRx = np.amax(outline, axis = 0)[0]
                LRy = np.amin(outline, axis = 0)[1]

                img_crop_polygon = newImArray[LRy:ULy, ULx:LRx]

                # reconvert to Image format for saving
                newIm = Image.fromarray(np.uint8(img_crop_polygon)) if (PIL.__version__ != '5.3.0') else Image.fromarray(img_crop_polygon).convert(mode="L")

                # save under output_path
                newIm.save(save_path)
        
    return


# crop_svg_outline("270.png", ID_dict = ID_dict, svg_coordinates = coord_list)
