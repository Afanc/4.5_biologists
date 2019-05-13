#!/usr/bin/python

import graph_representation as g_r

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
import numpy as np

#--------------------- all this shit could be done in a class - was lazy, didn't want to rewrite the whole thing
train_list = [line.rstrip('\n') for line in open('./data/train.txt')]
valid_list = [line.rstrip('\n') for line in open('./data/valid.txt')]

train_list = [i.split() for i in train_list]
valid_list = [i.split() for i in valid_list]

train_dic = {obj[0] : obj[1] for i, obj in enumerate(train_list)}
valid_dic = {obj[0] : obj[1] for i, obj in enumerate(valid_list)}

input_path = "data/gxl"

d=g_r.adj_matrix(input_path)

all_molecules = []

for k in d.keys() :
    if k[:-4] in [x[0] for x in train_list]:
        k = str(k)
        all_molecules.append(g_r.Molecule_object(k[:-4], train_dic[k[:-4]], d[k]))
    else :
        k = str(k)
        all_molecules.append(g_r.Molecule_object(k[:-4], valid_dic[k[:-4]], d[k]))

print(all_molecules[0].get_name())
#---------------------------------------------------
# list of molecule objects

a = [i.get_label() for i in all_molecules]
list_of_distances = [i.get_matrix() for i in all_molecules]
list_of_labels = [i.get_label() == 'a' for i in all_molecules]

list_of_labels = np.array(list_of_labels)

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(list_of_distances, list_of_labels)
