#!/usr/bin/python

import numpy as np
from PIL import Image, ImageDraw
import feature_extraction as fe

img = Image.open("270-25-05_Clock.png")

def scan_image_features(image_file, normalize_feature_matrix = False):
    """Scans image column-wise and returns vector of dimensions no_of_features x image width
    with features for each column (extracted by feature_extraction-function).
    Returns feature matrix for each image column, with option for matrix being normalized."""
    img = Image.open(image_file)
    img = img.convert("1")
    img_array = np.array(img)
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    no_of_features = len( fe.feature_extraction(img_array[:, 1].reshape(img_height, 1)) )  # get number of features assessed by feature_extraction-function
    feature_matrix = np.zeros(shape = (no_of_features, img_width))
    print("shape of feature matrix : ", feature_matrix.shape)
    for column in range(img_width):
        col = img_array[:, column].reshape(img_height, 1)
        col_features = fe.feature_extraction(col)
        feature_matrix[ : , column] = col_features
    if normalize_feature_matrix:
        feature_matrix = fe.normalization(feature_matrix)
    return feature_matrix

features_Clock = scan_image_features("270-25-05_Clock.png", normalize_feature_matrix=False)
#print(np.array(img).shape)
#print(features_Clock.shape)
# same as:
# features_Clock2 = scan_image_features("270-25-05_Clock.png")
# features_Clock_norm = scan_image_features("270-25-05_Clock.png", normalize_feature_matrix=True)
