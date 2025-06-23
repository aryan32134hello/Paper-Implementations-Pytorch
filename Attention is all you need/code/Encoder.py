import numpy as np
import torch
import torch.nn as nn
from self_attention import SelfAttention
from feed_forward import PositionWiseFeedForward

class Encoder(nn.Module):
    def __init__(self,embed_size,heads,ff_dim):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.ff_dim = ff_dim
        self.self_attention = SelfAttention(self.embed_size,heads)
        self.Feed_Forward = PositionWiseFeedForward(self.embed_size,self.ff_dim)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

    def forward(self,x,mask):
        attn_op = self.self_attention(x,x,x,mask) # here mask is use to hide the padded tokens as they dont have any information in it.
        layer_1_op = self.norm1(x+attn_op)
        feed_fwd_op = self.Feed_Forward(layer_1_op)
        layer_2_op = self.norm2(layer_1_op+feed_fwd_op)
        return layer_2_op
