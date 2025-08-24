import torch.nn as nn
from .utils.patch import PatchEmbeddings
from .utils.Transformer_Encoder import Tranformer_Encoder
from einops import rearrange

class VIT(nn.Module):

    def __init__(self, img_height, img_width, num_classes, channels, patch_size, num_heads, num_layers, ll_dim, dropout_rate, num_frames):
        super(VIT,self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.in_channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ll_dim = ll_dim
        self.dropout_rate = dropout_rate
        self.embed_dim = self.patch_size*self.patch_size*self.in_channels
        self.patch_embedding = PatchEmbeddings(self.img_height,self.img_width,self.in_channels,self.num_frames,self.embed_dim,self.patch_size)
        # self.encoder = Tranformer_Encoder(self.embed_dim,self.num_heads,self.ll_dim, self.dropout_rate) // for single layer
        self.encoder_layers = nn.ModuleList([
            Tranformer_Encoder(self.embed_dim,self.num_heads,self.ll_dim, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.norm_layer = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(in_features=self.embed_dim,out_features=self.num_classes)
    
    def forward(self,x):
        ## x -> B C T H W 
        B, C, T, H, W = x.shape
        x = self.patch_embedding(x) ## BT N D
        x = rearrange(x,'(b t) n d -> n (b t) d')
        # x = self.encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = rearrange(x, 'n (b t) d -> (b t) n d')
        norm_op = self.norm_layer(x)
        cls_token = norm_op[:,0]
        cls_token = rearrange(cls_token, '(b t) d -> b d t',b=B,t=T)
        
        cls_token = cls_token.unsqueeze(-1).unsqueeze(-1)
        return cls_token