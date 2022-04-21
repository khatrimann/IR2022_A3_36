#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 00:17:50 2022

@author: mann
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx



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
    
G = nx.from_numpy_matrix(adj_mat_undir, create_using=nx.DiGraph)

page_rank = nx.pagerank(G)
hubs, authorities = nx.hits(G)

compare = pd.DataFrame({"Node": list(idx_list.keys()), "Node_idx": list(idx_list.values()), "pageRank": list(page_rank.values()), "Hubs": list(hubs.values()), "Authorities": list(authorities.values())})