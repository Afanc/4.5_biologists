#!/usr/bin/python

import numpy as np
from scipy.spatial.distance import euclidean, cityblock
import dtw

def dyn_time_warp(word1, word2) :

    d, cost_matrix, acc_cost_matrix, path = dtw.dtw(word1, word2, dist=cityblock)
    #or dtw.accelerated_dtw(...) ?
    #or euclidean ?

    return (d, cost_matrix, acc_cost_matrix, path)

if(__name__=="__main__") :

    x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0])
    x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

    print("shape x", x.shape)

    d, cost_matrix, acc_cost_matrix, path = dtw.dtw(x, y, dist=euclidean)

    print(type(d), type(cost_matrix), type(acc_cost_matrix), type(path))
    print(d, cost_matrix, acc_cost_matrix, path)


# You can also visualise the accumulated cost and the shortest path
#import matplotlib.pyplot as plt
#
#plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
#plt.plot(path[0], path[1], 'w')
#plt.show()
