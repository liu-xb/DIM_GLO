import torch
import numpy as np
from .rerank import re_ranking
from sklearn.cluster import DBSCAN
import datetime
import random

def compute_mutual_knn(feature, mknn = 1, threshold = 0.5):
    t1 = datetime.datetime.now()
    dist = re_ranking(feature, k1=30, k2=6, lambda_value = 0.5)
    # index = np.argsort(dist, axis=1)
    t2 = datetime.datetime.now()

    print('compute eps')
    tri_mat = np.triu(dist,1)       # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
    tri_mat = np.sort(tri_mat,axis=None)
    top_num = np.round(1.6e-3*tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()
    print('eps',eps)
    t3 = datetime.datetime.now()

    cluster = DBSCAN(eps=eps,min_samples=4, metric='precomputed', n_jobs=8)
    
    t4 = datetime.datetime.now()

    labels = cluster.fit_predict(dist)

    pairs = []
    max_l = 0
    mean_l = 0
    for i in range(dist.shape[0]):
        pairs.append([])
        if labels[i] < 0:
            pairs[i].append(i)
            # for j in range(knn):
            #     if i in index[ index[i][j+1] ][1:knn+1]:
            #         if dist[i][index[i][j+1]] < threshold:
            #             pairs[i].append(index[i][j+1])
            continue
        for item in list(np.where(labels==labels[i])[0]):
            if dist[i][item] < threshold:
                pairs[i].append(item)
        # random.shuffle(pairs[i])
        temp_l = len(pairs[i])
        if temp_l > max_l:
            max_l = temp_l
        mean_l += temp_l
    t5 = datetime.datetime.now()

    print('max length: ', max_l)
    print('mean length: ', mean_l / 1. / dist.shape[0])
    print('rerank cost:',str(t2-t1))
    print('eps cost:', str(t3-t2))
    print('cluster cost:', str(t4-t3))
    print('pairs cost:', str(t5-t4))
    return pairs
