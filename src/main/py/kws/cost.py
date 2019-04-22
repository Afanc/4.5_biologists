import numpy as np
from matplotlib import pyplot as plt


#def euclidean(vector1, vector2):
#    """Uses numpy.linalg.norm to calculate the euclidean distance of two vectors."""
#    distance = np.linalg.norm(vector1-vector2, 2, 0) # the third argument "0" means the column, and "1" means the line.
#    return distance
# This one does the same as the euclidean function below - but the numpy.linalg.norm
# is somewhat a black box to me...

def euclidean(vector1, vector2):
    """Accepts two vectors of equal length (e.g. a column of a feature matrix
    corresponding to the features of a columns of an image) and calculates
    their Euclideam distance."""
    distance = np.sqrt(sum([(a - b)**2 for a, b in zip(vector1, vector2)]))  # list comprehension for squares of distances, summed up and square root taken
    return distance



def sakoe_chiba_band(features1, features2, band_width = 50):
    """Defines Sakoe-Chiba-band of width band_width around diagonal of 
    cost matrix for comparing feature matrices features1 and features2.
    Returns list of tuples with indices of the two feature matrices."""
    width1 = features1.shape[1]  # width of feature matrix 1 (features1)
    width2 = features2.shape[1]  # width of feature matrix 2 (features2)
    
    # define zero-matrix of appropriate size
    matrix_sc = np.zeros(shape = (width1, width2))
    # define diagonal
    steps = np.linspace(start = 0, stop = width1 - 1, num = width2)
    step_list = [int(i) for i in steps]

    # mark range around matrix diagonal with -1
    for i in range(width2):
        if step_list[i] - band_width >= 0:
            lower_bound = step_list[i] - band_width
        else:
            lower_bound = 0
        if step_list[i] + band_width <= width1:  # max. height
            upper_bound = step_list[i] + band_width
        else:
            upper_bound = width1
        for j in range(lower_bound, upper_bound):
            matrix_sc[j, i] = -1
    
    # get coordinates of the Sakoe-Chiba band
    band = np.where(matrix_sc < 0)
    band_coords = list(zip(band[0], band [1]))
    
    return band_coords


#### Example Sakoe -Chiba-band
#mat1 = np.zeros(shape = (4, 200))
#mat2 = np.zeros(shape = (4, 500))
#
#band_test = sakoe_chiba_band(mat1, mat2, 30)    
#
#matrix = np.full(shape = (mat1.shape[1], mat2.shape[1]), fill_value = 1000)
#for i in band_test:
#    matrix[i] = 0
#plt.imshow(matrix)



def dist_matrix(features1, features2, Sakoe_Chiba = True, band_width = 50):
    """ Creates distance matrix of comparison of two feature matrices, 
    e.g. output of scan_image_features-function.
    Offers Sakoe-Chiba-band option to consider paths only within band_width of 
    the matrix diagonal. If band_width reaches or exceeds width of feature matrix,
    the full comparison is calculated."""
    width1 = features1.shape[1]  # width of feature matrix 1 (features1)
    width2 = features2.shape[1]  # width of feature matrix 2 (features2)
    
    # initialize distance matrix matching the two feature matrices
    dist_matrix = np.full(shape = (width1, width2), fill_value = np.inf)

    # get tuples of indices of combinations of interest (i.e. within Sakoe-Chiba-band or complete matrix)
    if Sakoe_Chiba:
        band_coords = sakoe_chiba_band(features1, features2, band_width)
    else:
        band_coords = [(i, j) for i in np.arange(width1) for j in np.arange(width2)]  # list comprehension creating tuples of all matrix indices

    # calculate Euclidean distances of all combinations of interest
    dist = [euclidean(features1[:, i[0]], features2[:, i[1]]) for i in band_coords]    

    # enter distances into distance matrix
    for i, coords in enumerate(band_coords):
        dist_matrix[coords]  = dist[i]

    return dist_matrix


#test = dist_matrix(mat1, mat2, True, 10)
#test = dist_matrix(mat1, mat2, False)

# in the following line features1 and features2 are the output of 
# scan_image_features-function for two different images
#test = dist_matrix(features1, features2, True, 10)



###  I think next we need a function for determination of the minimal cost of the "alignment".

# def min_cost(dist_matrix):
    # ...
