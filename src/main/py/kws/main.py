#!/usr/bin/python

import os
import argparse
import crop_svg_outline as crop
import resize_images as res

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessing', default=True, type=bool)
parser.add_argument('--id_linking', default=True, type=bool)

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

exit()

for p in paths :
    if not os.path.exists(p):
        os.makedirs(p)

list_of_wordimages = os.listdir(paths["wordimages_input"])


#----- ID linking----#
if args.id_linking :

    for wordfile in list_of_wordimages :

        coord_list = extract_SVG_masks(word_file)

        test_dict = read_transcription(output = "word_dict")

        ID_dict = retrieve_IDs("General", test_dict)

        crop_svg_outline("270.jpg", ID_dict = ID_dict, svg_coordinates = coord_list)

        generate_word_ID_csv(word_dict = test_dict, file_name = "test_word_ID.txt", sep = ",")



#----- pre-processing ----#
if args.preprocessing :
    crop.crop_svg_outline("270.jpg", ID_dict = ID_dict, svg_coordinates = coord_list)

    exit()

    for image in list_of_images :

        crop.crop_svg_outline("270.jpg", ID_dict = ID_dict, svg_coordinates = coord_list)

        res.resize_images

        for file in list_of_words:
            img = plt.imread(input_path + '/' + file) #the loaded image has the shape: (height, widht, 4)

         #these steps remove the 4 channels in the 3. dimension
            frame = img[:, :, 3]
            img = img[:, :, 0]
            img = np.where((frame[:, :] == 0), 1, img[:, :])

            resized_img = cv2.resize(img, (100, 100)) #the reshaped image has the shape: (100, 100)

            scipy.misc.imsave(output_path + '/' + file, resized_img)

