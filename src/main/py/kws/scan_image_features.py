import numpy as np
from PIL import Image, ImageDraw

# img = Image.open("270-25-05_Clock.png")


def scan_image_features(image_file, normalize_matrix = False):
    """Scans image row-wise and returns vector of dimensions no_of_features x image width
    with features for each column."""
    img = Image.open(image_file)
    img = img.convert("1")
    img_array = np.array(img)
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    no_of_features = len( feature_extraction(img_array[:, 1].reshape(img_height, 1)) )  # get number of features assessed by feature_extraction-function
    feature_matrix = np.zeros(shape = (no_of_features, img_width))
    for column in range(img_width):
        col = img_array[:, column].reshape(img_height, 1)
        col_features = feature_extraction(col)
        feature_matrix[ : , column] = col_features
    if normalize_matrix:
        feature_matrix = normalization(feature_matrix)
    return feature_matrix

# features_Clock = scan_image_features("270-25-05_Clock.png", normalize_matrix=False)
# same as:
# features_Clock2 = scan_image_features("270-25-05_Clock.png")
# features_Clock_norm = scan_image_features("270-25-05_Clock.png", normalize_matrix=True)
