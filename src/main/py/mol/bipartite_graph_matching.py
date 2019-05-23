#!/usr/bin/python

import numpy as np

# implementation of the Hungarian algorithm
# https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.optimize.linear_sum_assignment.html
from scipy.optimize import linear_sum_assignment
# OR
# from munkres import Munkres, print_matrix   # http://software.clapper.org/munkres/

class BipartiteGraphMatching:
    def __init__(self, all_mol_dic):
        self.all_mol_dic = all_mol_dic

    # cost constants
    cost_subst = 4  # substitution = 1 del + 1 ins
    cost_indel = 2

    def calc_cost_matrix(self, mol1, mol2):
        bp1 = mol1.get_bipartite()
        bp2 = mol2.get_bipartite()
        n = len(bp1)
        m = len(bp2)
        c_size = n + m
        cost_matrix = np.full((c_size, c_size), 1e6)
        for (i, j), e in np.ndenumerate(cost_matrix):
            if i < n and j < m:
                cost_matrix[i, j] = (self.cost_subst if bp1[i][0] != bp2[j][0] else 0) + np.abs(bp1[i][1] - bp2[j][1])
            elif (i-n) == j or i == (j-m):
                cost_matrix[i, j] = self.cost_indel
            elif n < i and m < j:
                cost_matrix[i, j] = 0
        return cost_matrix

    def bp_edit_distance(self, mol1_id, mol2_id):   # not clear why here are coming floats
        mol1 = self.all_mol_dic[str(int(mol1_id))]
        mol2 = self.all_mol_dic[str(int(mol2_id))]
        cost_matrix = self.calc_cost_matrix(mol1, mol2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return cost_matrix[row_ind, col_ind].sum()


