#!/usr/bin/python

import numpy as np
from PIL import Image
import feature_extraction as fe

def scan_image_features(image_file, num_f = 9, normalize_feature_matrix = False):
    """Scans image column-wise and returns vector of dimensions no_of_features x image width
    with features for each column (extracted by feature_extraction-function).
    Returns feature matrix for each image column, with option for matrix being normalized."""
    no_of_features = num_f
    img = Image.open(image_file)
    img = img.convert("1")
    img_array = np.array(img)
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
#    no_of_features = len( fe.feature_extraction(img_array[:, 1].reshape(img_height, 1), num_f) )  # get number of features assessed by feature_extraction-function by extracting features of one colum 
    feature_matrix = np.zeros(shape = (no_of_features, img_width))
    #print("shape of feature matrix : ", feature_matrix.shape)
    for column in range(img_width):
        col = img_array[:, column].reshape(img_height, 1)
        col_features = fe.feature_extraction(col, no_of_features)
        feature_matrix[ : , column] = col_features
    if normalize_feature_matrix:
        feature_matrix = fe.normalization(feature_matrix, no_of_features)
    return feature_matrix

#features_new = scan_image_features("270-01-02_Letters.png", num_f = 9, normalize_feature_matrix = True)
#features_new = scan.scan_image_features("Test.png", num_f = 9, normalize_feature_matrix = True)
