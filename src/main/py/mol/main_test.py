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
train_list += [line.rstrip('\n') for line in open('./data/valid.txt')]
train_list = [i.split() for i in train_list]

train_dic = {obj[0]: obj[1] for i, obj in enumerate(train_list)}

input_path = os.path.join("data", "gxl")

all_mol_dic = g_r.adj_matrix(input_path)

# ------ training ---------------------------------------
# https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
train_ids = []
train_labels = []
for id, label in train_dic.items():
    train_ids.append([int(id)])
    train_labels.append(label)

print('\nTraining with %d samples ...' % len(train_ids))

bpm = BipartiteGraphMatching(all_mol_dic)

classifier = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric=bpm.bp_edit_distance)
classifier.fit(train_ids, train_labels)


# ----- testing -----------------------------------------

test_input_path = os.path.join("test-data", "gxl")

test_mol_dic = g_r.adj_matrix(test_input_path)

test_ids = []
for id in test_mol_dic.keys():
    test_ids.append([int(id)])

all_mol_dic = {**all_mol_dic, **test_mol_dic}

print('\nTesting with %d samples ...' % len(test_ids))

predictions = classifier.predict(test_ids)

# ----- save in report format -----------------------------------------
print('\nReporting predictions ...')

report_file_name = os.path.join("test-data", "predicted.txt")

with open(report_file_name, 'w') as fr:
    for key, predicted_class in np.ndenumerate(predictions):
        fr.write("{},{}\n".format(test_ids[key[0]][0], predicted_class))
