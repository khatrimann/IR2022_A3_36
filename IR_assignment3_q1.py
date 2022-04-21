#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 17:12:11 2022

@author: mann
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_num_conn(idx, adj_mat):
    num_conn = 0.
    nbr_idxs = np.where(adj_mat_undir[idx, :]==1)[0]
    for i in range(len(nbr_idxs)):
        for j in range(i+1, len(nbr_idxs)):
            if adj_mat[nbr_idxs[i], nbr_idxs[j]] == 1:
                num_conn += 1.
                
    return num_conn

data = pd.read_csv("/home/mann/Downloads/soc-sign-bitcoinalpha.csv", header=None)
data.columns = ['source', 'target', 'rating', 'time']

edge_list = []
for node in data.source.unique():
    tmp_df = data[data.source==node]
    tmp_edge_list = np.column_stack((tmp_df.source,tmp_df.target))
    edge_list = edge_list + tmp_edge_list.tolist()
    
idx_list = {val: idx for idx, val in enumerate(np.unique(np.append(np.sort(data.source.unique()), np.sort(data.target.unique()))))}
                                               
adj_mat_dir = np.zeros((len(idx_list), len(idx_list)))
adj_mat_undir = np.zeros((len(idx_list), len(idx_list)))

for nodes in edge_list:
    adj_mat_dir[idx_list[nodes[0]], idx_list[nodes[1]]] = 1
    
    adj_mat_undir[idx_list[nodes[0]], idx_list[nodes[1]]] = 1
    adj_mat_undir[idx_list[nodes[1]], idx_list[nodes[0]]] = 1
    
# Dataset statistics
# Nodes 	3783
# Edges 	24,186 (taking bi-directional edges as two different edges) else it will be 24186-20124=4062
# Avg in-degree 6.393338620142744
# Avg out degree 6.393338620142744
# Max in degree 490
# Max out degree 398
# Density = E/N*(N-1) = 24186 / 3783*3782 = 24179.606661379858

# OUT_NODES -> rows
# IN_NODES -> columns

total_nodes = len(idx_list)
out_degrees = np.sort(np.sum(adj_mat_dir, axis=1))
in_degrees = np.sort(np.sum(adj_mat_dir, axis=0))

out_avg = np.sum(np.sum(adj_mat_dir, axis=1))/len(idx_list)
in_avg = np.sum(np.sum(adj_mat_dir, axis=0))/len(idx_list)

out_nodes_dist = { out_deg: (out_degrees.tolist()).count(out_deg)/total_nodes for out_deg in np.unique(out_degrees)}
in_nodes_dist = { out_deg: (out_degrees.tolist()).count(out_deg)/total_nodes for out_deg in np.unique(in_degrees)}

plt.bar(in_nodes_dist.keys(), in_nodes_dist.values())
plt.xlabel("Degree")
plt.ylabel("Fraction of nodes")
plt.title("In Degree Distribution")

plt.figure()

plt.bar(out_nodes_dist.keys(), out_nodes_dist.values())
plt.xlabel("Degree")
plt.ylabel("Fraction of nodes")
plt.title("Out Degree Distribution")

coeff = {}
undir_degrees = []
for node, idx in idx_list.items():
    degree = np.where(adj_mat_undir[idx, :]==1)[0].shape[0]
    undir_degrees.append(degree)
    if degree not in [0, 1]:
        num_conn = get_num_conn(idx, adj_mat_undir)
        coeff[node] = ((2*num_conn)/(degree * (degree - 1)))
    else:
        coeff[node] = 0
        
undir_degrees = np.unique(np.sort(undir_degrees))
undir_degrees = undir_degrees[1:]
        
# avg clus coeff on each degree
avg_clus = [sum(coeff)/degree for degree in undir_degrees]

plt.figure()

plt.plot(undir_degrees, avg_clus)
plt.xlabel("Degree")
plt.ylabel("Average over degree")
plt.title("Degree Distribution")
