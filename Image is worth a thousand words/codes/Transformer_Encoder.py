import torch
import torch.nn as nn
import numpy as np
from multi_head_attention import MultiHeadAttention
from multilayer_perceptron import MultiLayerPerceptron

class Tranformer_Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ll_dim, dropout_rate):
        super(Tranformer_Encoder,self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ll_dim = ll_dim
        self.dropout_rate = dropout_rate 
        self.norm_layer = nn.LayerNorm(self.embed_dim)
        self.multi_head_attention = MultiHeadAttention(self.num_heads,self.embed_dim)
        self.multi_layer_perceptron = MultiLayerPerceptron(self.ll_dim,self.embed_dim,self.dropout_rate)

    def forward(self, x):
        norm_op_1 = self.norm_layer(x)
        attn_op = self.multi_head_attention(norm_op_1,norm_op_1,norm_op_1)
        sum_op_1 = attn_op + x
        norm_op_2 = self.norm_layer(sum_op_1)
        mlp_op = self.multi_layer_perceptron(norm_op_2)
        final_sum_op = mlp_op + sum_op_1
        return final_sum_op