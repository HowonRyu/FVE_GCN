import torch
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from FVE_GCN_utils import *

import glob
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time

from distutils.util import strtobool
import argparse


#from torch_geometric.nn import TopKPooling
#from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
#from torch_geometric.utils import (add_self_loops, sort_edge_index, remove_self_loops)
#from torch_sparse import spspmm
#from net.braingraphconv import MyNNConv


if None:   # from BrainGNN Li et al. -- no longer use
    class Network(torch.nn.Module):
        def __init__(self, indim, ratio, nclass=1, k=8, R=3):
            '''
            :param indim: (int) node feature dimension
            :param ratio: (float) pooling ratio in (0,1)
            :param nclass: (int)  number of classes
            :param k: (int) number of communities
            :param R: (int) number of ROIs
            '''
            super(Network, self).__init__()

            self.indim = indim
            self.dim1 = 32
            self.dim2 = 32
            self.dim3 = 512
            self.dim4 = 256
            self.dim5 = 8
            self.k = k
            self.R = R

            #self.n1 = nn.Sequential(nn.Linear(self.indim, self.dim1),nn.ReLU(), nn.Linear(self.dim1, self.dim1))
            self.n1 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1 * self.indim))
            self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
            self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
            self.n2 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1))
            self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
            self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

            #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
            #self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
            self.fc1 = torch.nn.Linear(2 * (self.dim1 + self.dim2), self.dim2)
            self.bn1 = torch.nn.BatchNorm1d(self.dim2)
            self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
            self.bn2 = torch.nn.BatchNorm1d(self.dim3)
            self.fc3 = torch.nn.Linear(self.dim3, nclass)



        def forward(self, x, edge_index, batch, edge_attr, pos):
            x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr, pseudo=pos)
            x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)

            pos = pos[perm]
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            edge_attr = edge_attr.squeeze()
            edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

            x = self.conv2(x, edge_index, edge_attr, pos)
            x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)

            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = torch.cat([x1,x2], dim=1)
            x = self.bn1(F.relu(self.fc1(x)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.bn2(F.relu(self.fc2(x)))
            x= F.dropout(x, p=0.5, training=self.training)
            x = torch.sigmoid(self.fc3(x))  ### softmax eliminate, force to [0,1]

            #return x, self.pool1.weight, self.pool2.weight, torch.sigmoid(score1).view(x.size(0),-1), torch.sigmoid(score2).view(x.size(0),-1)
            return x

        def augment_adj(self, edge_index, edge_weight, num_nodes):
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                    num_nodes=num_nodes)
            edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                    num_nodes)
            edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                            edge_weight, num_nodes, num_nodes,
                                            num_nodes)
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            return edge_index, edge_weight






class GCN_new(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim1=256, hidden_dim2=128, hidden_dim3=32, pool_ratio=0.6):
        super(GCN_new, self).__init__()
        
        # GCN + pooling layers
        self.conv1 = GCNConv(num_node_features, hidden_dim1, bias=True)
        self.pool1 = TopKPooling(hidden_dim1, ratio=pool_ratio)

        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, bias=True)
        self.pool2 = TopKPooling(hidden_dim2, ratio=pool_ratio)

        self.conv3 = GCNConv(hidden_dim2, hidden_dim3, bias=True)
        self.pool3 = TopKPooling(hidden_dim3, ratio=pool_ratio)

        # classifier after readout for regression
        self.linear = torch.nn.Linear(hidden_dim3 * 2, 1)

        # initialization
        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        #readout
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.linear(x)


class GCN_new2(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, hidden_dim4=32, pool_ratio=0.6):
        super(GCN_new, self).__init__()
        
        # GCN + pooling layers
        self.conv1 = GCNConv(num_node_features, hidden_dim1, bias=True)
        self.pool1 = TopKPooling(hidden_dim1, ratio=pool_ratio)

        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, bias=True)
        self.pool2 = TopKPooling(hidden_dim2, ratio=pool_ratio)

        self.conv3 = GCNConv(hidden_dim2, hidden_dim3, bias=True)
        self.pool3 = TopKPooling(hidden_dim3, ratio=pool_ratio)

        self.conv4 = GCNConv(hidden_dim3, hidden_dim4, bias=True)
        self.pool4 = TopKPooling(hidden_dim4, ratio=pool_ratio)

        # classifier after readout for regression
        self.linear = torch.nn.Linear(hidden_dim4 * 2, 1)

        # initialization
        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv4.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)

        #readout
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.linear(x)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim1=32, hidden_dim2=15):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim1, bias=True)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, bias=True)
        self.batch_norm1 = torch.nn.LayerNorm(hidden_dim1)
        self.batch_norm2 = torch.nn.LayerNorm(hidden_dim2)
        self.linear = torch.nn.Linear(hidden_dim2, 1)

        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index)), negative_slope=0.01)
        x = global_mean_pool(x, batch)
        return self.linear(x)


class GCN_bare(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 1, bias=False)
        self.post_pool = torch.nn.Linear(1, 1, bias=True)

        torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
        torch.nn.init.xavier_uniform_(self.post_pool.weight)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = global_add_pool(x, batch)
        return self.post_pool(x)


class GCN_deeper(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(64, 1)

        # Initialize weights (uniform)
        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)

        x = global_mean_pool(x, batch) 

        return self.linear(x) 




def debug_check(batch):
    if torch.isnan(batch.x).any():
        print("NaN detected in X")
    if torch.isinf(batch.x).any():
        print("Inf detected in X")
    if torch.isnan(batch.y).any():
        print("NaN detected in Y")
    if torch.isinf(batch.y).any():
        print("Inf detected in Y")
    if batch.edge_index.max() >= batch.x.shape[0]:
        print("Edge index out of bounds:", batch.edge_index.max().item(), "vs", batch.x.shape[0])



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        debug_check(batch)


        out = model(batch.x, batch.edge_index, batch.batch)

        target = batch.y.view(-1, 1).float()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)



@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)  
        out = model(batch.x, batch.edge_index, batch.batch)
        
        target = batch.y.view(-1, 1).float()  
        loss = criterion(out, target)
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model, train_loader, test_loader, epochs=100, patience=999, lr=0.001, weight_decay=0.0001, log=None, device='cpu'):
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} {time.ctime(time.time())}", file=log, flush=True)


        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1


        if (patience != 999) & (patience_counter >= patience):
            print(f"Early stopping triggered after {epoch} epochs (best test loss: {best_loss:.6f})", file=log, flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best model with test loss:", best_loss, file=log, flush=True)


    return model



def main(args):
    if args.partial_dat:
        dat_type = "GCN_partial"
    elif args.partial_tsa_dat:
        dat_type = "GCN_partial_tsa"
    else:
        if args.icld_age_sex:
            dat_type = "GCN_cov"
        else:
            dat_type = "GCN"


    job_full_nickname = f"{args.job_nickname}_{dat_type}_{args.model_type}_lr{args.lr}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if (args.partial_dat and args.icld_age_sex) or (args.partial_tsa_dat and args.icld_age_sex):
        raise ValueError("Incompatible configuration: cannot use partial data with covariates.")

    with open(f"GCN_models/{job_full_nickname}.txt", "w") as f:
        print(f"Writing model output... using device: {device}", file=f, flush=True)
        print("{}".format(args).replace(', ', ',\n'))
        #print(
        #    f"epochs={args.epochs}, lr={args.lr}, wd={args.weight_decay}, patience={args.patience}, batch_size={args.batch_size}, "
        #    f"norm_y={args.norm_y}, icld_age_sex={args.icld_age_sex}, partial={args.partial_dat}, partial_tsa={args.partial_tsa_dat}, "
        #    f"scaler={args.scaler_name}, device={device}",
        #    file=f,
        #    flush=True,
        #)

        # data load
        print(f"Loading data at {time.ctime(time.time())}", file=f, flush=True)

        if args.partial_dat:
            FVE_df_org = pd.read_csv("data/FVE_dat_partial.csv")
            SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
            non_surface_area_vars = ["nihtbx_cryst_uncorrected"]
            dat_type = "GCN_partial"
        elif args.partial_tsa_dat:
            FVE_df_org = pd.read_csv("data/FVE_dat_partial_tsa.csv")
            SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
            non_surface_area_vars = ["nihtbx_cryst_uncorrected"]
            dat_type = "GCN_partial_tsa"
        else:
            FVE_df_org = pd.read_csv("data/FVE_dat.csv")
            SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
            non_surface_area_vars = ["interview_age", "sex_2", "nihtbx_cryst_uncorrected"]
            if args.icld_age_sex:
                dat_type = "GCN_cov"
            else:
                dat_type = "GCN"
        
        print(f"input sanity check: partial={args.partial_dat}, partial_tsa={args.partial_tsa_dat}, x_cols={FVE_df_org.columns[0:1]} to {FVE_df_org.columns[20483:len(FVE_df_org.columns)]}")

        train_loader, val_loader, test_loader = input_to_graph(
            SurfeView_surfaces=SurfeView_surfaces,
            FVE_df_all=FVE_df_org,
            partial_dat=args.partial_dat,
            partial_tsa_dat=args.partial_tsa_dat,
            scaler=args.scaler_name,
            norm_y=args.norm_y,
            icld_age_sex=args.icld_age_sex,
            batch_size=args.batch_size,
        )



        print(f"Data loaded at {time.ctime(time.time())}", file=f, flush=True)

        # select model
        if args.icld_age_sex:
            num_node_features = 3
        else:
            num_node_features = 1

        print(f"num_node_features={num_node_features}")
        if args.model_type == "base":
            model = GCN_new(num_node_features=num_node_features).to(device)
        elif args.model_type == "base2":
            model = GCN_new2(num_node_features=num_node_features).to(device)
        elif args.model_type == "deep":
            model = GCN_deeper(num_node_features=num_node_features).to(device)
        elif args.model_type == "BrainGNN":
            model = Network(indim=num_node_features, ratio=0.8).to(device)

        # training
        print(f"Start training at {time.ctime(time.time())}", file=f, flush=True)
        boolean_map = {False: "F", True: "T"}


        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            log=f,
            device=device,
        )

        # model save
        torch.save(
            trained_model,
            f"GCN_models/{job_full_nickname}.pt",
        )
        print(f"Start eval at {time.ctime(time.time())}", file=f, flush=True)

        # eval
        trained_model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = trained_model(batch.x, batch.edge_index, batch.batch)
                y_pred.extend(out.cpu().numpy().flatten())
                y_true.extend(batch.y.cpu().numpy().flatten())

        y_pred, y_true = np.array(y_pred), np.array(y_true)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print(f"R2 Score: {r2:.4f}, MSE: {mse:.4f}", file=f, flush=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--job_nickname", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="base",
                        choices=["base", "deep", "BrainGNN", "new", "base2"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--norm_y", action="store_true")
    parser.add_argument("--partial_dat", action="store_true")
    parser.add_argument("--partial_tsa_dat", action="store_true")
    parser.add_argument("--icld_age_sex", action="store_true")
    parser.add_argument("--scaler_name", type=str, default="standard",
                        choices=["standard", "minmax"],)

    args = parser.parse_args()
    main(args)
