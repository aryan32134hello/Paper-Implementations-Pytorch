import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class PatchEmbeddings(nn.Module):
    
    def __init__(self, img_height, img_width, in_channels,num_frames, embed_dim, patch_size = 16):
        super(PatchEmbeddings,self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = int((self.img_height*self.img_width)/(patch_size**2))
        self.embed_size = embed_dim
        self.num_frames = num_frames
        self.conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.embed_size,
                              kernel_size=self.patch_size,stride=self.patch_size)
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embed_size),requires_grad=True)
        self.positional_encoding = nn.Parameter(torch.rand(1,self.num_patches+1,self.embed_size),requires_grad=True)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames,self.embed_size))

    def forward(self,x):

        conv_output = self.conv(x)
        flatten_op = self.flatten(conv_output) # N, n_channel_op, 196
        flatten_op = flatten_op.transpose(1,2) # N, 196, n_channel_op
        cls_token = self.cls_token.expand(flatten_op.shape[0],-1,-1) 
        conv_with_cls = torch.cat([flatten_op,cls_token],dim = 1) ## bt, 197, D
        final_op = conv_with_cls + self.positional_encoding
        n = final_op.shape[1]
        final_op = rearrange(final_op,'(b t) n d -> (b n) t d',t=self.num_frames)
        final_op = final_op + self.temporal_embedding
        final_op = rearrange(final_op,'(b n) t d -> (b t) n d',n = n)

        return final_op ## BT N D
  
        