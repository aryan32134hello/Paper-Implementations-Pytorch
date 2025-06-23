import torch
import torch.nn as nn
import numpy as np
from self_attention import SelfAttention
from feed_forward import PositionWiseFeedForward

class Decoder(nn.Module):
    def __init__(self,embed_size,head,ff_dim):
        super(Decoder,self).__init__()
        self.embed_size = embed_size
        self.head = head
        self.ff_dim = ff_dim
        self.masked_multiheadattention = SelfAttention(self.embed_size,self.head)
        self.multi_headattention = SelfAttention(self.embed_size,self.head)
        self.feed_forward = PositionWiseFeedForward(self.embed_size,self.ff_dim)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        self.norm3 = nn.LayerNorm(self.embed_size)

    def forward(self,x,encoder_op,input_mask, output_mask):
        masked_attn_op = self.masked_multiheadattention(x,x,x,mask = output_mask) # mask not implemented
        sub_layer1_op = self.norm1(x+masked_attn_op)
        attn_op = self.multi_headattention(encoder_op,encoder_op,sub_layer1_op,mask = input_mask) # here mask is use to hide the padded tokens as they dont have any information in it.
        sub_layer2_op = self.norm2(attn_op+sub_layer1_op)
        feed_fwd_op = self.feed_forward(sub_layer2_op)
        sub_layer3_op = self.norm3(feed_fwd_op+sub_layer2_op)
        return sub_layer3_op

