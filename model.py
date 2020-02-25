import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class Zhang(nn.Module):
    def __init__(self, user_num, item_num, aspect_num, embed_size1, embed_size2):
        super().__init__()
        
        self.embed_u1 = nn.Embedding(user_num, embed_size1)
        self.embed_u2 = nn.Embedding(user_num, embed_size2)
        self.embed_i1 = nn.Embedding(item_num, embed_size1)
        self.embed_i2 = nn.Embedding(item_num, embed_size2)
        self.lin = nn.Linear(embed_size1, aspect_num)

        
    def forward(self, user_id, item_id):
        u_v1 = self.embed_u1(user_id)
        u_v2 = self.embed_u2(user_id)
        i_v1 = self.embed_u1(item_id)
        i_v2 = self.embed_u2(item_id)
        
        pred_asp_u = self.lin(u_v1)
        pred_asp_i = self.lin(i_v1)
        
        prob = torch.cat([i_v1, i_v2], dim=1) * torch.cat([u_v1, u_v2], dim=1)
        prob = torch.sigmoid(torch.sum(prob, dim=1))
        
        #out = torch.sigmoid(out.view(batch_size))
        
        return prob, pred_asp_u, pred_asp_i
    
    def predict(self, user_id, item_id):
        u_v1 = self.embed_u1(user_id)
        u_v2 = self.embed_u2(user_id)
        i_v1 = self.embed_u1(item_id)
        i_v2 = self.embed_u2(item_id)
        
        prob = torch.cat([i_v1, i_v2], dim=1) * torch.cat([u_v1, u_v2], dim=1)
        prob = torch.sigmoid(torch.sum(prob, dim=1))
        
        return prob