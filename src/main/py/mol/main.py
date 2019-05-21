#!/usr/bin/python

import graph_representation as g_r
from bipartite_graph_matching import BipartiteGraphMatching

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# implementation of the Hungarian algorithm
# https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.optimize.linear_sum_assignment.html
from scipy.optimize import linear_sum_assignment
# OR
# from munkres import Munkres, print_matrix   # http://software.clapper.org/munkres/

# working directory should be "src/main/py/mol"

# --------------------- all this shit could be done in a class - was lazy, didn't want to rewrite the whole thing
train_list = [line.rstrip('\n') for line in open('./data/train.txt')]
valid_list = [line.rstrip('\n') for line in open('./data/valid.txt')]

train_list = [i.split() for i in train_list]
valid_list = [i.split() for i in valid_list]

train_dic = {obj[0]: obj[1] for i, obj in enumerate(train_list)}
valid_dic = {obj[0]: obj[1] for i, obj in enumerate(valid_list)}

input_path = os.path.join("data", "gxl")

all_mol_dic = g_r.adj_matrix(input_path)

# so far not needed
# all_molecules = []
#
# for k in all_mol_dic.keys() :
#     if k[:-4] in [x[0] for x in train_list]:
#         k = str(k)
#         all_molecules.append(g_r.Molecules(k[:-4], train_dic[k[:-4]], all_mol_dic[k]))
#     else :
#         k = str(k)
#         all_molecules.append(g_r.Molecules(k[:-4], valid_dic[k[:-4]], all_mol_dic[k]))
#
# print(all_molecules[0].get_name())
# ---------------------------------------------------
# list of molecule objects

# a = [i.get_label() for i in all_molecules]
# list_of_distances = [i.get_matrix() for i in all_molecules]
# list_of_labels = [i.get_label() == 'a' for i in all_molecules]
#
# list_of_labels = np.array(list_of_labels)

# ------ training ---------------------------------------
# https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
train_ids = []
train_labels = []
for id, label in train_dic.items():
    # still don't need
    # mol = all_molecules[id]
    # or
    # file = id + '.gxl'
    # mol = g_r.Molecules(file)
    train_ids.append([int(id)])
    train_labels.append(label)

print('\nTraining with %d samples ...' % len(train_ids))

bpm = BipartiteGraphMatching(all_mol_dic)

classifier = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric=bpm.bp_edit_distance)
classifier.fit(train_ids, train_labels)


# ----- validating -----------------------------------------

valid_ids = []
valid_labels = []
for id, label in valid_dic.items():
    valid_ids.append([int(id)])
    valid_labels.append(label)

print('\nValidating with %d samples ...' % len(valid_ids))

predictions = classifier.predict(valid_ids)


# ----- accuracy -----------------------------------------

accuracy = accuracy_score(valid_labels, predictions) * 100
print('\nThe accuracy of OUR classifier is %d%%' % accuracy)

# ----- save in report format -----------------------------------------
report_file_name = os.path.join("data", "predicted.txt")

with open(report_file_name, 'w') as fr:
    for key, predicted_class in np.ndenumerate(predictions):
        fr.write("{}[{}],{}\n".format(valid_ids[key[0]][0], valid_labels[key[0]], predicted_class))
