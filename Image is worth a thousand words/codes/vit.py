import torch
import torch.nn as nn
import numpy as np
from patch import PatchEmbeddings
from Transformer_Encoder import Tranformer_Encoder

class VIT(nn.Module):

    def __init__(self, img_height, img_width, num_classes, channels, patch_size, num_heads, num_layers, ll_dim, dropout_rate):
        super(VIT,self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.in_channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ll_dim = ll_dim
        self.dropout_rate = dropout_rate
        self.embed_dim = self.patch_size*self.patch_size*self.in_channels
        self.patch_embedding = PatchEmbeddings(self.img_height,self.img_width,self.in_channels,self.embed_dim,self.patch_size)
        # self.encoder = Tranformer_Encoder(self.embed_dim,self.num_heads,self.ll_dim, self.dropout_rate) // for single layer
        self.encoder_layers = nn.ModuleList([
            Tranformer_Encoder(self.embed_dim,self.num_heads,self.ll_dim, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.norm_layer = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(in_features=self.embed_dim,out_features=self.num_classes)
    
    def forward(self,x):
        x = self.patch_embedding(x)
        # x = self.encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        norm_op = self.norm_layer(x)
        cls_token = norm_op[:,0,:]
        classifier_op = self.classifier(cls_token)
        return classifier_op