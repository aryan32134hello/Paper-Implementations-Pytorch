import torch
import torch.nn as nn
import numpy as np
import math

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert (self.head_dim*heads==embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries = nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size,bias=False)

    
    def split_heads(self,X):

        batch_size, token_length, embed_size = X.shape()
        X = X.reshape(batch_size,token_length,self.heads,self.head_dim)
        return X

    def scaled_dot_product_attention(self,query,keys,values,mask=None):

        attn_score = torch.matmul(query,keys.transpose(2,3))/math.sqrt(self.head_dim)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask==0,-1e9)
        attn_prob = torch.softmax(attn_score,dim=-1)
        final_attn_score = torch.matmul(attn_prob,values)
        return final_attn_score

    def combine_heads(self,X):

        batch_size,heads, token_length, head_dim = X.shape()
        X = X.transpose(1,2)
        return X.reshape(batch_size,token_length,self.embed_size)

        ## view needs the tensor should be contiguous but transpose make it non-contigous so we should use contigous() before view 
        # but not in reshape() 

    def forward(self,values,keys,query,mask=None):

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        ##mask not implemented
        ## SPLIT THE MATRIX INTO HEADS

        values = self.split_heads(values)
        keys = self.split_heads(keys)
        query = self.split_heads(query)
        ## above the shape is going to be (batch_size, token_length, heads, head_dim) 
        ## but we want heads to be seperate so that it can multiply each heads independently 
        ## so what we do is we take a transpose on axis 1 and 2
        values = values.transpose(1,2)
        keys = keys.transpose(1,2)
        query = query.transpose(1,2)
        ## now shape -> (batch_size, heads, token_length, head_dim)
        attn_output = self.scaled_dot_product_attention(self,query,keys,values)
        ## attn_output shape -> (batch_size, heads, token_length, head_dim)
        output = self.combine_heads(attn_output)
        output = self.fc_out(output)
        
        return output

        
        
        


