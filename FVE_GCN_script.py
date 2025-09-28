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

import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
#from torch_sparse import spspmm
#from net.braingraphconv import MyNNConv


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from distutils.util import strtobool



if None:
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



class GCN_deeper(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(64, 1)

        # Initialize weights (optional, but good)
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


def save_model(model, path="GCN_models/model.pt"):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        debug_check(batch)


        out = model(batch.x, batch.edge_index, batch.batch)

        target = batch.y.view(-1).float()   
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
        
        target = batch.y.view(-1).float()   # MSE requires float
        loss = criterion(out, target)
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model, train_loader, test_loader, epochs=100, patience=5, lr=0.001, weight_decay=0.0001, log=None, device='cpu'):
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"lr={lr}, weight_decay={weight_decay}")
    print(f"lr={lr}, weight_decay={weight_decay}", file=log, flush=True)


    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} {time.ctime(time.time())}")
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} {time.ctime(time.time())}", file=log, flush=True)

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs (best test loss: {best_loss:.6f})")
            print(f"Early stopping triggered after {epoch} epochs (best test loss: {best_loss:.6f})", file=log, flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best model with test loss:", best_loss)
        #log.write(f"Restored best model with test loss: {best_loss}\n")
        #log.flush()

    return model


def main(lr, weight_decay, epochs, job_nickname=None, model_type="base", patience=5, partial=False):
    job_full_nickname = f"{job_nickname}_p{patience}_lr{lr}_wd{weight_decay}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    with open(f"/niddk-data-central/mae_hr/FVE/CGN_models/{job_full_nickname}.txt", "w") as f:
        f.write("Writing model output... \n")
        f.write(f"Using device: {device} \n")
        f.flush()
        batch_size = 100

        print(f"Loading data at {time.ctime(time.time())}", file=f, flush=True)
        print(f"partia={partial}", file=f, flush=True)
        if partial:
            graph_list = torch.load("/niddk-data-central/mae_hr/FVE/CGN_data/FVE_CGN_graph_list_partial.pt")
        else:
            graph_list = torch.load("/niddk-data-central/mae_hr/FVE/CGN_data/FVE_CGN_graph_list.pt")

        train_dataset, test_dataset = train_test_split(graph_list, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        print(f"Data loaded at {time.ctime(time.time())}", file=f, flush=True)
        
        # define model
        if model_type == "base":
            model = GCN(num_node_features=1).to(device)  
        elif model_type == "deep":
            model = GCN_deeper(num_node_features=1).to(device)  
        elif model_type == "BrainGNN":
            model = Network(indim=1, ratio=0.8).to(device)

        # model training
        print(f"Start training at {time.ctime(time.time())}", file=f, flush=True)
        trained_model = train_model(model=model, train_loader=train_loader, test_loader=test_loader,
                                    lr=lr, weight_decay=weight_decay, epochs=epochs, patience=5, log=f, device=device)
        
        #save_model(trained_model, f"/niddk-data-central/mae_hr/FVE/CGN_models/{job_full_nickname}_weights.pt")
        torch.save(trained_model, f"/niddk-data-central/mae_hr/FVE/CGN_models/{job_full_nickname}_full.pt")


        print(f"Start eval at {time.ctime(time.time())}", file=f, flush=True)
 

        trained_model.eval()
        y_pred, y_true = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                if model_type == "BrainGNN":
                    out = trained_model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos)
                else:
                    out = trained_model(batch.x, batch.edge_index, batch.batch)
                y_pred.extend(out.cpu().numpy().flatten())  
                y_true.extend(batch.y.cpu().numpy().flatten())  

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print(f'R2 Score: {r2:.4f}, MSE: {mse:.4f}')
        print(f'R2 Score: {r2:.4f}, MSE: {mse:.4f}', file=f, flush=True)



if __name__ == "__main__":
    lr = float(sys.argv[1]) 
    weight_decay = float(sys.argv[2])
    epochs = int(sys.argv[3])
    job_nickname = sys.argv[4]
    model_type = str(sys.argv[5])
    patience = int(sys.argv[6])
    partial = bool(strtobool(sys.argv[7]))

    main(lr=lr, weight_decay=weight_decay, epochs=epochs, job_nickname=job_nickname, model_type=model_type, patience=patience, partial=partial)
