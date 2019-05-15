#!/usr/bin/python
import xml.etree.ElementTree as ET
import os
import random
import numpy as np
import pandas as pd

# implementation of the Hungarian algorithm
# https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.optimize.linear_sum_assignment.html
from scipy.optimize import linear_sum_assignment
# OR
# from munkres import Munkres, print_matrix   # http://software.clapper.org/munkres/

# working directory should be "src/main/py/mol"

cost_subst = 3
cost_indel = 3

class Molecule_object:
    def __init__(self, name, label, adj_matrix) :
        self.name = name
        self.label = label 
        self.adj_matrix = adj_matrix
    
    def get_name(self):
        return self.name
    def get_label(self):
        return self.label
    def get_matrix(self):
        return self.adj_matrix

class Molecules:
    def __init__(self, filename):

        self.filename = "data/gxl/" + filename
        self.tree = ET.parse(self.filename) # with ET.parse we can read in the gxl file
        self.node = [node.text.strip(" ") for node in self.tree.findall(".//node/attr/string")] #gets the lable of all the nodes
        self.edge_values = [edge.text for edge in self.tree.findall(".//edge/attr/int")] #gets the value of each edge (corresponding to the number of bonds between two atoms)

        # creats a nested list
        # for each edge the two corresponding nodes will be extracted from the file saved in a nested list.
        # e.g [[0,2],[0,5],[0,7].....]   --> means node 0 is connected with node 2, 5, and 7
        # we subtract by -1 because we want to use the nodes as indices later--> e.g.  the first node should be node 0
        self.start_end = [[int(edge.get('from').strip("_"))-1, int(edge.get('to').strip("_"))-1] for edge in self.tree.findall(".//edge")] #


    def get_adj_matrix(self):

        # creats an empty adjacency matrix in form of a nested list
        # size of the matrix is equal to the number of nodes --> len(self.node)
        adjacency_matrix = [[0 for i in range(len(self.node))] for j in range(len(self.node))]

        # iterates over the node pairs connected by edges
        for counter, node_pair in enumerate(self.start_end):
            #fills the edge values of the corresponding node pairs into the adjacency matrix
            adjacency_matrix[node_pair[0]][node_pair[1]] = int(self.edge_values[counter])
            adjacency_matrix[node_pair[1]][node_pair[0]] = int(self.edge_values[counter])
        # inserts the node lables into the diagonal of the adjacency matrix
        for i in range(len(self.node)):
            adjacency_matrix[i][i] = self.node[i]
        return adjacency_matrix

    def get_bipartite(self):
        adjm = self.get_adj_matrix()
        bplen = adjm.shape[0]
        bp = np.array((bplen, 2))
        for i in range(bplen):
            bp[i, 0] = adjm[i, i]   # nodes: atoms
            bp[i, 1] = sum(np.apply_along_axis(int, 0, adjm[i]))   # edges: covalent bonds


def adj_matrix(folder_of_gxl_files):
    """
    :param folder_of_gxl_files: folder which contains the gxl files
    :return: dictionary with filename as key and adjacency matrix as value
    Additionally instances for all the gxl files are created
    example: instance M16 is created for file 16.gxl
    """
    # get all .gxl files
    list_of_molecules = os.listdir(folder_of_gxl_files)

    d = {}
    # iterate over every .gxl file
    for file in list_of_molecules:
        file_num = int(file.strip(".gxl"))

        # creats an instance for a molecule
        # example for file 16.gxl:
        # globals()["M%s" % file_num] = Molecules(16.gxl) would correspond to M16 = Molecules(16.gxl)
        globals()["M%s" % file_num] = Molecules(file)

        # add key value pair to dictionary. Key is the filename (e.g. "16.gxl") and the value is the adjacency Matrix for this file
        d[file] = globals()["M%s" % file_num].get_adj_matrix()

    return d


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


def graph_distance_edit(mol1, mol2):
    # TODO prepare cost matrix for x/y, node substitutions costs (atoms dissimilarity, covalent-bonds dissimilarity)
    # TODO run Hungarian algorithm to estimate the cost (= graph distance edit(x,y))
    cost_matrix = calc_cost_matrix(mol1, mol2)
    d = linear_sum_assignment(cost_matrix)
    return random.uniform(0., 1.)


#######################################################################
if __name__ == "__main__":
    # working directory should be "src/main/py/mol"
    input_path = "data/gxl"

    #apply function to create instances and to get dictionary containing the adjacency matrix
    d=adj_matrix(input_path)

    #print(adjacency matrix for instance M16 (corresponding to file 16.gxl)
    print("adjacency matrix for 16.gxl")
    print(M16.get_adj_matrix(), "\n")

    # print adjacency matrix for Molecule 9770
    print("adjacency matrix for 9770.gxl, nice representation")
    for i in d["9770.gxl"]:
        print(i)


