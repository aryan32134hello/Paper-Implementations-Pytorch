import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self,embed_size,out):
        super(PositionWiseFeedForward,self).__init__()

        self.input_feature = embed_size
        self.output_1 = out
        self.linear_1 = nn.Linear(in_features=self.input_feature,out_features=self.output_1)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=self.output_1,out_features=self.input_feature)
    
    def forward(self,input):
        
        output = self.linear_2(self.activation(self.linear_1(input)))
        return output
