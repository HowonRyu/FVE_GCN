import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import glob
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time
from torch_geometric.nn import NNConv as MyNNConv, TopKPooling
import scipy.io
import scipy.sparse as sp
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)


def load_surface_mesh(SurfeView_surfaces):
    # coords
    coords_lh = SurfeView_surfaces['surf_lh_pial'][0,0]['vertices'][:10242]
    coords_rh = SurfeView_surfaces['surf_rh_pial'][0,0]['vertices'][:10242]

    # faces
    faces_lh = SurfeView_surfaces['icsurfs'][0,5]['faces'][0,0] - 1
    faces_rh = SurfeView_surfaces['icsurfs'][0,5]['faces'][0,0] - 1 + 10242

    # concatenate for left and right
    coords = np.concatenate([coords_lh, coords_rh])
    faces  = np.concatenate([faces_lh, faces_rh])
    return coords, faces




def build_surface_adjacency(coords, faces):
    """
    Build an adjacency list (or adjacency matrix) from surface mesh faces.
    
    Args:
        coords (ndarray): (n_vertices, 3) array of vertex coordinates
        faces (ndarray): (n_faces, 3) array, each row has 3 vertex indices forming a triangular face
    
    Returns:
        adjacency_matrix (ndaray): sparse adjacency_matrix is a set of vertices adjacent to vertex v
    """
    n_vertices = coords.shape[0]
    adjacency = [set() for _ in range(n_vertices)]

    # For each face, connect its three vertices (v1, v2, v3)
    for (v1, v2, v3) in faces:
        adjacency[v1].add(v2)
        adjacency[v1].add(v3)
        adjacency[v2].add(v1)
        adjacency[v2].add(v3)
        adjacency[v3].add(v1)
        adjacency[v3].add(v2)

    n_vertices = len(adjacency)
    rows = []
    cols = []
    for v in range(n_vertices):
        for w in adjacency[v]:
            rows.append(v)
            cols.append(w)
    data = np.ones(len(rows), dtype=int)
    
    # Build a square adjacency matrix
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    
    # Symmetrize (sometimes needed if faces produce directed edges—but typically it's already symmetric)
    # A = A.maximum(A.T)
    
    return A





def input_to_graph(SurfeView_surfaces, FVE_df_all, norm_y=False, scaler="minmax",
                   partial_dat=False, icld_age_sex=False, batch_size=20):
    vertices, faces = load_surface_mesh(SurfeView_surfaces)

    # graph structure
    A_csr = build_surface_adjacency(vertices, faces)
    edge_index = torch.tensor(np.array(A_csr.nonzero()), dtype=torch.long)


    # clean FVE_df of no NAs
    FVE_df = FVE_df_all.dropna()


    # train, val, test split
    train_valid, test = train_test_split(FVE_df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_valid, test_size=0.2, random_state=42)

    if partial_dat:
        norm_cols = list(set(FVE_df.columns) - set('nihtbx_cryst_uncorrected'))
    else:
        norm_cols = list(set(FVE_df.columns) - set(['nihtbx_cryst_uncorrected', 'sex_2']))

    X_train = train[norm_cols]
    X_val   = val[norm_cols]
    X_test  = test[norm_cols]

    y_train = train["nihtbx_cryst_uncorrected"]
    y_val   = val["nihtbx_cryst_uncorrected"]
    y_test  = test["nihtbx_cryst_uncorrected"]


    # standardization
    if scaler=="minmax":
        scaler_X = MinMaxScaler().fit(X_train)
    else:
        scaler_X = StandardScaler().fit(X_train)

    
    X_train_scaled_all = scaler_X.transform(X_train)
    X_val_scaled_all   = scaler_X.transform(X_val)
    X_test_scaled_all  = scaler_X.transform(X_test)

    # standardize y
    if norm_y:
        if scaler=="minmax":
            scaler_y = MinMaxScaler().fit(y_train.values.reshape(-1,1))
        else:
            scaler_y = StandardScaler().fit(y_train.values.reshape(-1,1))
        y_train_scaled = scaler_y.transform(y_train.values.reshape(-1,1)).flatten()
        y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1,1)).flatten()
        y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1,1)).flatten()
    else:
        y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

    print(f"error check: {y_train_scaled.shape}")
    # Graph
    if partial_dat:
        train_graph = create_graph_data(X_all=X_train_scaled_all, y=torch.Tensor(y_train_scaled), partial_dat=partial_dat, 
                                edge_index = edge_index, vertices = torch.FloatTensor(vertices), icld_age_sex=False)
        validation_graph = create_graph_data(X_all=X_val_scaled_all, y=torch.Tensor(y_val_scaled), partial_dat=partial_dat, 
                                        edge_index = edge_index, vertices = torch.FloatTensor(vertices), icld_age_sex=False)
        test_graph = create_graph_data(X_all=X_test_scaled_all, y=torch.Tensor(y_test_scaled), partial_dat=partial_dat, 
                                        edge_index = edge_index, vertices = torch.FloatTensor(vertices), icld_age_sex=False)    
    else:
        train_graph = create_graph_data(X_all=X_train_scaled_all, y=torch.Tensor(y_train_scaled), sex=train['sex_2'], 
                                        partial_dat=partial_dat, 
                                edge_index = edge_index, vertices = torch.FloatTensor(vertices), icld_age_sex=icld_age_sex) 
        validation_graph = create_graph_data(X_all=X_val_scaled_all, y=torch.Tensor(y_val_scaled), sex=val['sex_2'],
                                             partial_dat=partial_dat, 
                                        edge_index = edge_index, vertices = torch.FloatTensor(vertices), icld_age_sex=icld_age_sex)
        test_graph = create_graph_data(X_all=X_test_scaled_all, y=torch.Tensor(y_test_scaled), sex=test['sex_2'],
                                       partial_dat=partial_dat, 
                                        edge_index = edge_index, vertices = torch.FloatTensor(vertices), icld_age_sex=icld_age_sex)    

    # Loaders
    train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(validation_graph, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_graph, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


def create_graph_data(X_all, y, edge_index, vertices, sex=None, partial_dat=False, icld_age_sex=False):
    graphs = []
    
    #pos
    pos = vertices
    
    #edge_attr
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1, keepdim=True)
    
    #node features: X, interview_age, and sex
    if partial_dat:
        X = torch.tensor(X_all, dtype=torch.float32).unsqueeze(-1)
    else:
        X = torch.tensor(X_all[:,:-1], dtype=torch.float32).unsqueeze(-1)

    if icld_age_sex:
        interview_age = torch.tensor(X_all[:,-1])
        sex = torch.tensor(sex.values)
        age_expanded = interview_age.unsqueeze(1).repeat(1, X.shape[1]).unsqueeze(2)
        sex_expanded = sex.unsqueeze(1).repeat(1, X.shape[1]).unsqueeze(2)           
        node_feat = torch.tensor(torch.cat([X, age_expanded, sex_expanded], dim=2), dtype=torch.float32)
    else:
        node_feat = torch.tensor(X, dtype=torch.float32)

    print(f"icld_age_sex = {icld_age_sex}, node_feat.shape: {node_feat.shape}")
    for i in range(node_feat.shape[0]):        
        data = Data(
            x=node_feat[i],
            edge_index=edge_index,
            pos = pos,
            #edge_attr = edge_attr,
            y= torch.FloatTensor(y[i])
        )
        graphs.append(data)
    return graphs



