#!/usr/bin/python

import numpy as np
from scipy.optimize import linear_sum_assignment


class BipartiteGraphMatching:
    def __init__(self, all_mol_dic):
        self.all_mol_dic = all_mol_dic

    # cost constants
    cost_subst = 3
    cost_indel = 3

    def calc_cost_matrix(self, mol1, mol2):
        bp1 = mol1.get_bipartite()
        bp2 = mol2.get_bipartite()
        n = len(bp1)
        m = len(bp2)
        c_size = n + m
        cost_matrix = np.zeros((c_size, c_size))
        for (i, j), e in np.ndenumerate(cost_matrix):
            if i < n and j < m:
                cost_matrix[i, j] = (self.cost_subst if bp1[i][0] != bp2[j][0] else 0) + np.abs(bp1[i][1] - bp2[j][1])
            elif (i-n) == j or i == (j-m):
                cost_matrix[i, j] = self.cost_indel
        return cost_matrix

    def bp_edit_distance(self, mol1_id, mol2_id):   # not clear why here are coming floats
        mol1 = self.all_mol_dic[str(int(mol1_id))]
        mol2 = self.all_mol_dic[str(int(mol2_id))]
        cost_matrix = self.calc_cost_matrix(mol1, mol2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return cost_matrix[row_ind, col_ind].sum()


