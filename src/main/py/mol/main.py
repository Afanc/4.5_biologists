#!/usr/bin/python

import graph_representation as g_r

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import numpy as np
import csv

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

train_dic = {obj[0] : obj[1] for i, obj in enumerate(train_list)}
valid_dic = {obj[0] : obj[1] for i, obj in enumerate(valid_list)}

input_path = "data/gxl"

d=g_r.adj_matrix(input_path)

# all_molecules = []

# for k in d.keys() :
#     if k[:-4] in [x[0] for x in train_list]:
#         k = str(k)
#         all_molecules.append(g_r.Molecules(k[:-4], train_dic[k[:-4]], d[k]))
#     else :
#         k = str(k)
#         all_molecules.append(g_r.Molecules(k[:-4], valid_dic[k[:-4]], d[k]))

# print(all_molecules[0].get_name())
# ---------------------------------------------------
# list of molecule objects

cost_subst = 3
cost_indel = 3


def calc_cost_matrix(mol1, mol2):
    bp1 = mol1.get_bipartite()
    bp2 = mol2.get_bipartite()
    c_size = len(bp1) + len(bp2)
    cost_matrix = np.zeros((2*c_size, 2*c_size))
    for (i, j), e in np.ndenumerate(cost_matrix):
        if i < c_size or j < c_size:
            cost_matrix[i, j] = cost_subst if bp1[i, 0] != bp2[j, 0] else 0
            cost_matrix[i, j] += np.abs(bp2[i, 1] != bp2[j, 1])
        elif (i-c_size) == j or i == (j-c_size):
            cost_matrix[i, j] = cost_indel
    return cost_matrix


def graph_distance_edit(mol1_id, mol2_id):
    # TODO prepare cost matrix for x/y, node substitutions costs (atoms dissimilarity, covalent-bonds dissimilarity)
    # TODO run Hungarian algorithm to estimate the cost (= graph distance edit(x,y))
    mol1 = globals()["M%s" % mol1_id]
    mol2 = globals()["M%s" % mol2_id]
    cost_matrix = calc_cost_matrix(mol1, mol2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()


all_molecules = g_r.adj_matrix('./data/gxl')
train_molecules = np.array([])
train_labels = np.array([])
for id, label in train_dic.items():
    mol = all_molecules[id]
    # or
    #file = id + '.gxl'
    #mol = g_r.Molecules(file)  # globals()["M%s" % id]
    np.append(train_molecules,  np.array([id, id]))
    np.append(train_labels, label)


# a = [i.get_label() for i in all_molecules]
# list_of_distances = [i.get_matrix() for i in all_molecules]
# list_of_labels = [i.get_label() == 'a' for i in all_molecules]
#
# list_of_labels = np.array(list_of_labels)

classifier = KNeighborsClassifier(n_neighbors=5, metric=graph_distance_edit)
classifier.fit(train_molecules, train_labels)

# pred = classifier.predict(test_molecules)

#  OR
#classifier = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', metric='pyfunc', func=graph_distance_edit)
#classifier.fit(train_molecules, train_labels)
