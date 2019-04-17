#!/usr/bin/python

import os
import argparse
import crop_svg_outline as crop
import resize_images as res
import read_SVG
import read_transcription as read_trans
import crop_svg_outline as c_s_o

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

#directories
paths["images_input"] = 'data/images'
paths["images_output"] = 'data/binarized_images'
paths["wordimages_input"] = "data/word_images"
paths["wordimages_output"] = "data/resized_word_images"
paths["svg"] = "data/ground-truth/locations"

for p in paths :
    if not os.path.exists(p):
        os.makedirs(p)

#files
paths["transcription.txt"] = "data/ground-truth/transcription.txt"

#adapt if run from 4.5_biologists
if(os.getcwd()[-14:] == "4.5_biologists"):
    for k in paths :
        paths[k] = "src/main/py/kws/" + paths[k]



list_of_wordimages = os.listdir(paths["wordimages_input"])
list_of_svg = os.listdir(paths["svg"])


#----- ID linking----#
if args.id_linking :

    for filename in list_of_svg:
        svgfile= paths["svg"]+"/"+filename
        imagefile=paths["images_input"]+"/"+filename[:-4]+".jpg"

        coord_list = read_SVG.extract_SVG_masks(svgfile)

        word_dic = read_trans.read_transcription(file_name=paths["transcription.txt"], output = "word_dict")

        ID_dict = read_trans.retrieve_IDs("General", word_dic)

#this does not work for whatever reason
        c_s_o.crop_svg_outline(imagefile, ID_dict = ID_dict, svg_coordinates = coord_list)

        generate_word_ID_csv(word_dict = word_dic, file_name = "word_IDs.txt", sep = ",")

        break

exit()


#----- pre-processing ----#
if args.preprocessing :

    for image in list_of_images :

        res.resize_images
#blabla
