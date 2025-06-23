import torch
import torch.nn as nn
import numpy as np
from self_attention import SelfAttention
from Positional_Encoding import PositionalEncoding
from feed_forward import PositionWiseFeedForward
from Encoder import Encoder
from Decoder import Decoder
from torch.nn import Embedding

# In transformers paper it says to share weights of Input_Embedding, Output_embedding and linear_layer before softmax
# It can be done only if ip and op embed. have same size and same kind of words like both english, not possible in 
# english to french translation
# here I am sharing weights as original paper  

class Transformers(nn.Module):
    def __init__(self,vocab_size,embed_size,head,ff_dim,seq_len):
        super(Transformers,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.heads = head
        self.ff_dim = ff_dim
        self.seq_len = seq_len
        self.embedding = Embedding(self.vocab_size,self.embed_size)
        self.positional_encoding = PositionalEncoding(self.seq_len,self.embed_size)
        self.encoder = Encoder(self.embed_size,self.heads,self.ff_dim) ## here Nx = 1
        self.decoder = Decoder(self.embed_size,self.heads,self.ff_dim)
        self.linear = nn.Linear(self.embed_size,self.output_vocab_size)

        ##sharing weights and embeddings
        self.linear.weight = self.embedding.weight
        self.input_embedding = self.embedding
        self.output_embedding = self.embedding

    def generate_masks(self, input, output):

        input_mask = (input!=0).unsqueeze(1).unsqueeze(2)
        output_mask = (output!=0).unsqueeze(1).unsqueeze(3) # here unsqueeze(3) happens because it has to AND with upper_triangular mask for future tokens
        ## why upper triangular matrix is used here ##
        ## Lets take an example of seq_len = 4 so after multiplication of query and key it gives (4,4) as output 
        ##   0 1 2 3
        ## 0 1 0 0 0 here query i can only attend till key i not (i>j) so that value will be future and will be 
        ## 1 1 1 0 0 treated as zero
        ## 2 1 1 1 0
        ## 3 1 1 1 1

        hide_future_mask = (1-torch.triu(torch.ones(self.seq_len, self.seq_len),diagonal=1)).bool()
        output_mask = output_mask & hide_future_mask
        return input_mask,output_mask

    def forward(self,input,output):
        ip_embed = self.positional_encoding(self.input_embedding(input))
        op_embed = self.positional_encoding(self.output_embedding(output))
        input_mask, output_mask = self.generate_masks(input,output)
        encoder_op = self.encoder(ip_embed,input_mask)
        decoder_op = self.decoder(op_embed,encoder_op,input_mask,output_mask)
        linear_op = self.linear(decoder_op)
        probab = torch.softmax(linear_op,dim=-1)

        return probab




