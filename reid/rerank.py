import numpy as np
from scipy.spatial.distance import cdist
import datetime

def re_ranking(feat,k1,k2,lambda_value, MemorySave = False, Minibatch = 2000):

    query_num = feat.shape[0]
    all_num = query_num
    # feat = np.append(probFea,galFea,axis = 0)
    # feat = feat.astype(np.float32)
    print('computing original distance')
    if MemorySave:
        original_dist = np.zeros(shape = [all_num,all_num],dtype = np.float32)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it,] = np.power(cdist(feat[i:it,],feat),2).astype(np.float32)
            else:
                original_dist[i:,:] = np.power(cdist(feat[i:,],feat),2).astype(np.float32)
                break
            i = it
    else:
#        original_dist = cdist(feat,feat).astype(np.float32)  
#        original_dist = np.power(original_dist,2).astype(np.float32)
        original_dist = 2 - 1.98 * np.dot(feat, np.transpose(feat)) #* 0.99
    del feat    
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)
   
    print('starting re_ranking')
    t1 = datetime.datetime.now()
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    # original_dist = original_dist[:query_num,]    
    t2 = datetime.datetime.now()
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])
    
    t3 = datetime.datetime.now()
    
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    
    t4 = datetime.datetime.now()
    
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    # final_dist = final_dist[:query_num,query_num:]
    print(str(t2-t1))
    print(str(t3-t2))
    print(str(t4-t3))
    return final_dist
