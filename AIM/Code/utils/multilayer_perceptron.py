import torch
import torch.nn as nn
import numpy as np 
import math

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, ll_dim, embed_dim, dropout_rate):
        super(MultiLayerPerceptron,self).__init__()
        self.ll_dim = ll_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.ll_1 = nn.Linear(self.embed_dim,self.ll_dim)
        self.qgelu = QuickGELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.ll_2 = nn.Linear(self.ll_dim,self.embed_dim)
    
    def forward(self, x):

        ll_1_op = self.ll_1(x)
        gelu_op = self.qgelu(ll_1_op)
        ll_2_op = self.ll_2(self.dropout(gelu_op))
        op = self.dropout(ll_2_op)
        return op