import torch.nn as nn
import torch
import numpy as np


class Adapters(nn.Module):
    
    def __init__(self, d_model, hidden_ratio, skip_connect = True):
        super(Adapters,self).__init__()

        self.d_model = d_model
        self.hidden_ratio = hidden_ratio
        self.skip_connect = skip_connect
        self.hidden_feature = hidden_ratio*self.d_model
        self.fc1 = nn.Linear(self.d_model,self.hidden_feature)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_feature,self.d_model)

    def forward(self, x):

        xs = self.fc1(x)
        xs = self.act_fn(xs)
        xs = self.fc2(xs)

        if self.skip_connect:
            x = x + xs 
        else :
            x = xs

        return x 