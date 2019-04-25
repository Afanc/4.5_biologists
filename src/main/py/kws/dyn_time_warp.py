#!/usr/bin/python

from scipy.spatial.distance import euclidean, cityblock
import dtw


def dyn_time_warp(word1, word2):
    d, cost_matrix, acc_cost_matrix, path = dtw.dtw(word1, word2, dist=cityblock)
    #or dtw.accelerated_dtw(...) ?
    #or euclidean ?

    return (d, cost_matrix, acc_cost_matrix, path)

