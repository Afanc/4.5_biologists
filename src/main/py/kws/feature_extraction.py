#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

def feature_extraction(col, num_f):
    """
    input: column of a image
    output: feature vector containing the different feature values of this column

    features:
        number_of_black_pixels
        upper_boundary
        lower_boundary
        black_white_transitions
        ...?
    """
    # update this if you add a feature
    num_of_features = num_f

    # number of black pixels
    number_of_black_pixels = np.sum(col == 0)
    #if there are black pixels in this column the other features need to be calculated
    if number_of_black_pixels > 0:
        upper_boundary = np.argwhere(col == 0)[0][0]  # get first index of black pixel
        lower_boundary = np.argwhere(col == 0)[-1][0]   # get index of last black pixel (using -1 for reverse slicing)
        black_white_transitions = 0
        for row in range(len(col)-1):  # iterate over column
            # when there is a transition from black to white or reverse, the counter is increased by one
            if col[row] != col[row+1]:
                black_white_transitions += 1
    # if there are no black pixels the features get the following values
    else:
        upper_boundary = len(col)
        lower_boundary = len(col)
        black_white_transitions = 0  # this will cause errors on a pure black or pure white image, but as soon as there is a single transition in one columns of the image, it should work
    feature_values = np.array([number_of_black_pixels, upper_boundary, lower_boundary, black_white_transitions]).reshape(num_of_features, )
    return feature_values


def get_feature_vectors(img):
    """
    takes an image as input
    computes the feature vectors for the different features
    returns a matrix of the feature vectors (each row corresponds to a feature vector
    """
    feature_matrix = np.zeros(shape = (4, img.shape[1]))
    for col in range(img.shape[1]):
        feature_matrix[:, col] = feature_extraction(img[:,col].reshape(img.shape[0],1))
    return feature_matrix


def normalization(feature_matrix):
    """
    takes a feature matrix as imput: each row corresponds to the feature vector of a image for one particular feature
    returns a normalized feature matrix  (xi-mean)/sd
    """
    for row in range(4):
        feature_matrix[row, :] = (feature_matrix[row, :] - np.mean(feature_matrix[row, :])) / np.std(feature_matrix[row,:])
    return feature_matrix

if(__name__=='__main__') :
    pass
#short example
#img = plt.imread("data/resized_word_images/270-01-01_s_2.png")
#plt.imshow(img)
#plt.show()
#a=get_feature_vectors(img)
#print("image shape ", np.array(img).shape)
##print(a[:,:20])
#print("feature matrix shape : ", a.shape)
#n=normalization(a)
#print("normalized feature matrix shape : ", n.shape)
##print(n[:,:20])
