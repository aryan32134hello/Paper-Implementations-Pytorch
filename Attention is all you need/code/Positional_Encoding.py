import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self,seq_len,embed_size):
        super(PositionalEncoding,self).__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.pe = torch.zeros(self.seq_len,self.embed_size)
        # for pos in range(self.pe.shape[0]):
        #     for i in range(self.pe.shape[1]):
        #         if i%2==0:
        #             self.pe[pos,i] = torch.sin(torch.tensor(pos)/(10000**(i/self.embed_size)))
        #         else:
        #             self.pe[pos,i] = torch.cos(torch.tensor(pos)/(10000**(i/self.embed_size)))
        # self.pe = self.pe.unsqueeze(0)
        position = torch.arange(0,self.seq_len,dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0,self.embed_size,2)*-(np.log(10000)/self.embed_size))
        self.pe[:,0::2] = torch.sin(position*div)
        self.pe[:,1::2] = torch.cos(position*div)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,x):
        # print("x shape:", x.shape)
        # print("pe slice shape:", self.pe[:, :x.size(1), :].shape)
        x = x + self.pe[:,:x.size(1),:] ## here it can happen that X has smaller max_len so we can slice the pe till max_len of X and then add it 
        return x

