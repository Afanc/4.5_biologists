import numpy as np
from PIL import Image, ImageDraw

# img = Image.open("270-25-05_Clock.png")


def scan_image_features(image_file, no_of_features):
    """Scans image row-wise and returns vector of dimensions no_of_features x image width
    with features for each column."""
    img = Image.open(image_file)
    img = img.convert("1")
    img_array = np.array(img)
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    feature_vector = np.zeros(shape = (no_of_features, img_width))  #, dtype = uint8)
    for column in range(img_width):
        col = img_array[:, column].reshape(img_height, 1)
        col_features = feature_extraction(col)
        feature_vector[ : , column] = col_features
    return feature_vector

# features_Clock = scan_image_features("270-25-05_Clock.png", 4)
