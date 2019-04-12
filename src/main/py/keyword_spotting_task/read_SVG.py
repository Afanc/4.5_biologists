import re


def extract_SVG_masks(file_name):
    with open(file_name, "r") as data:
        coord_data = list()
        for line in data:
            ID = re.search("\d{3}\-\d{2}\-\d{2}", line)
            coords = re.search("M\s\d(.*?)\d\sZ", line)
            if ID and coords:  # i.e. a line contains both
                ID_coords = (ID.group(0), coords.group(0))
                coord_data.append(ID_coords)
    return coord_data

# coord_list = extract_SVG_masks("270.svg")
