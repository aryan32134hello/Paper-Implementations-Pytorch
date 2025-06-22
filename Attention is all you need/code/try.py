import torch

max_seq_length = 100
position = torch.arange(0, max_seq_length, dtype=torch.float)
print(position.shape)
print(position)
position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
print(position.shape)
print(position)
div_term = torch.arange(0,512,2)
print(div_term.shape)
print(div_term)
mul = position*div_term
print(mul)
print(mul.shape)