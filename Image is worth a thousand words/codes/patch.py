import torch
import torch.nn as nn
import numpy as np

class PatchEmbeddings(nn.Module):
    
    def __init__(self, img_height, img_width, in_channels, embed_dim, patch_size = 16):
        super(PatchEmbeddings,self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = int((self.img_height*self.img_width)/(patch_size**2))
        self.embed_size = embed_dim
        self.conv = nn.Conv2d(in_channels=self.in_channels,out_channels=self.embed_size,
                              kernel_size=self.patch_size,stride=self.patch_size)
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embed_size),requires_grad=True)
        self.positional_encoding = nn.Parameter(torch.rand(1,self.num_patches+1,self.embed_size),requires_grad=True)

    def forward(self,x):

        conv_output = self.conv(x)
        flatten_op = self.flatten(conv_output)
        flatten_op = flatten_op.transpose(1,2)
        cls_token = self.cls_token.expand(flatten_op.shape[0],-1,-1)
        conv_with_cls = torch.cat([flatten_op,cls_token],dim = 1)
        final_op = conv_with_cls + self.positional_encoding

        return final_op

        