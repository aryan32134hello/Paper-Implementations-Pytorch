import torch
import torch.nn as nn
import numpy as np
import math

class MultiHeadAttention(nn.Module,):
    def __init__(self,num_heads,embed_dim):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = int(self.embed_dim/self.num_heads)

        self.Q_l = nn.Linear(self.embed_dim,self.embed_dim)
        self.K_l = nn.Linear(self.embed_dim,self.embed_dim)
        self.V_l = nn.Linear(self.embed_dim,self.embed_dim)

        self.fc_out = nn.Linear(self.embed_dim,self.embed_dim)

    def split_head(self,x):
        batch_size,token_len,embed_dim = x.shape()
        x = x.reshape(batch_size,token_len,self.num_heads,self.head_dim)
        return x
    
    def self_attention_product(self,key,query,value):

        qk_product = torch.matmul(query,key.transpose(2,3))/math.sqrt(self.head_dim)
        softmax_op = torch.softmax(qk_product)
        final_product = torch.matmul(softmax_op,value)
        return final_product

    def combine_heads(self, x):

        batch_size, num_heads, token_len, head_dim = x.shape()
        x = x.transpose(1,2)
        x = x.reshape(batch_size,token_len,self.embed_dim)
        return x

    def forward(self,key,query,value):

        ### k,q,v has shape (batch, token_len, embed_dim)

        key = self.K_l(key)
        query = self.Q_l(query)
        value = self.V_l(value)

        key = self.split_head(key).transpose(1,2)
        query = self.split_head(query).transpose(1,2)
        value = self.split_head(value).transpose(1,2)

        ## after split head and transpose it has shape (batch, num_heads, token_len, head_dim)

        attn_output = self.self_attention_product(key,query,value)

        ## now it has shape of (batch, num_heads, token_len, head_dim)
        ## we have to make it to original shape

        final_op = self.combine_heads(attn_output)

        ## now it has shape of (batch, token_len, embed_dim)

        final_op = self.fc_out(final_op)
        return final_op