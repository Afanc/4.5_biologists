import scipy.misc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

input_path = r"C:\Users\svenw\Desktop\GitHub\4.5_biologists\src\main\py\kws\data\word_images"
output_path = r"C:\Users\svenw\Desktop\GitHub\4.5_biologists\src\main\py\kws\data\resized_word_images"


if not os.path.exists(output_path):
    os.makedirs(output_path)

list_of_words = os.listdir(input_path)

for file in list_of_words:
    img = plt.imread(input_path + '/' + file)

    resized_img = cv2.resize(img, (100, 100))

    scipy.misc.imsave(output_path + '/' + file, resized_img)