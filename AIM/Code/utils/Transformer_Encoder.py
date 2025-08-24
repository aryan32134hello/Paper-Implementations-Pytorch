import torch
import torch.nn as nn
import numpy as np
from multi_head_attention import MultiHeadAttention
from multilayer_perceptron import MultiLayerPerceptron
from Adapters import Adapters
from einops import rearrange

class Tranformer_Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ll_dim, dropout_rate, num_frames, scale):
        super(Tranformer_Encoder,self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ll_dim = ll_dim
        self.dropout_rate = dropout_rate 
        self.hidden_ratio = 0.25
        self.num_frames = num_frames
        self.norm_layer = nn.LayerNorm(self.embed_dim)
        self.norm_layer_2 = nn.LayerNorm(self.embed_dim)
        self.multi_head_attention = MultiHeadAttention(self.num_heads,self.embed_dim)
        self.multi_layer_perceptron = MultiLayerPerceptron(self.ll_dim,self.embed_dim,self.dropout_rate)
        self.T_Adapter = Adapters(self.embed_dim,self.hidden_ratio,False)
        self.S_Adapter = Adapters(self.embed_dim,self.hidden_ratio,False)
        self.MLP_Adapter = Adapters(self.embed_dim,self.hidden_ratio,False)

    def forward(self, x):
        ## x -> n bt d
        xt = rearrange(x,'n (b t) d -> t (b n) d',t = self.num_frames)
        xt = self.norm_layer(xt)
        xt = self.multi_head_attention(xt,xt,xt) ## T-MSA
        xt = self.T_Adapter(xt)
        xt = rearrange(xt,'t (b n) d -> n (b t) d',)
        x = x + xt
        x_1 = self.norm_layer(x)
        attn_op = self.multi_head_attention(x_1,x_1,x_1)
        attn_op = self.S_Adapter(attn_op)
        sum_op_1 = attn_op + x
        norm_op_2 = self.norm_layer_2(sum_op_1)
        mlp_op = self.multi_layer_perceptron(norm_op_2)
        xm = self.MLP_Adapter(norm_op_2)

        final_sum_op = mlp_op + sum_op_1 + xm
        return final_sum_op