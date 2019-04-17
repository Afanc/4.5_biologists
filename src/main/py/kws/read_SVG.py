#!/usr/bin/python

import re
import numpy as np

def extract_SVG_masks(file_name):
    """Reads svg-file and returns list of ID, xy-coordinates (as numpy-array and list of xy-tuples) and svg-coordinates (as string)."""
    with open(file_name, "r") as data:
        coord_data = list()
        for line in data:
            ID = re.search("\d{3}\-\d{2}\-\d{2}", line)
            coords_svg = re.search("M\s\d(.*?)\d\sZ", line)
            if ID and coords_svg:  # i.e. a line contains both an ID and coordinate information
                coords_svg = coords_svg.group(0)
                coords_xy = re.sub("[A-Z]\s", "", coords_svg)  # remove leading and internal svg-identifiers
                coords_xy = re.sub("\s[A-Z]$", "", coords_xy)  # remove trailing svg-identifier
                coords_separated = [int(float(value)) for value in coords_xy.split()]  # separates x and y-coordinates
                coords_xy_array = np.array(list(zip(coords_separated[0::2], coords_separated[1::2])))  # places x- and y-coordinates in two-column numpy-array
                coords_xy_tuple = list(zip(coords_separated[0: :2], coords_separated[1: :2]))
                ID_coords = (ID.group(0), coords_xy_array, coords_xy_tuple, coords_svg)
                coord_data.append(ID_coords)
    return coord_data

# coord_list = extract_SVG_masks("270.svg")
