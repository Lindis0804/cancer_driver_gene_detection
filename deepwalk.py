import numpy as np
import pandas as pd
import time
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops,add_self_loops

from common import device
import sys

data = torch.load("./data/CPDB_data.pkl")
model =  Node2Vec(data.edge_index, embedding_dim=16, walk_length=80,
                     context_size=5,  walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)
loader = model.loader(batch_size=128, shuffle=True)
TRAINING_TIMES = 10
CHECK_POINT_FILE = 'checkpoint.txt'

def save_checkpoint(epoch, model, optimizer, filename=CHECK_POINT_FILE):
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, filename)

# Function to load model and optimizer state
def load_checkpoint(model, optimizer, filename=CHECK_POINT_FILE):
  try:
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    _epoch = checkpoint['epoch']
  except:
    _epoch = 1
  return model, optimizer, _epoch

model, optimizer, _epoch = load_checkpoint(model,optimizer)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    save_checkpoint(epoch,model,optimizer)
    print("finish traing.")
    return total_loss / len(loader)

for epoch in range(_epoch, TRAINING_TIMES):
    print(f'training in time {epoch}...')
    loss = train()
    print (loss)

model.eval()
str_fearures = model()

torch.save(str_fearures, 'str_fearures.pkl')