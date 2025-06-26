import torch
import torch.nn as nn
import numpy as np
from Transformer import Transformers

## ------------------------------DUMMY TRAINING TO TEST FOR ERRORS-----------------------------------------------

VOCAB_SIZE = 10000
EMBED_SIZE = 512
HEADS = 8
FF_DIM = 2048
MAX_SEQ_LEN = 100

transformer_model = Transformers(VOCAB_SIZE,EMBED_SIZE,HEADS,FF_DIM,MAX_SEQ_LEN)

source_data = torch.randint(1,VOCAB_SIZE,(4,MAX_SEQ_LEN))
target_data = torch.randint(1,VOCAB_SIZE,(4,MAX_SEQ_LEN))

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(transformer_model.parameters(),lr=0.001,betas=(0.9, 0.98), eps=1e-9)

transformer_model.train()

for epoch in range(10):
    optim.zero_grad()
    output = transformer_model(source_data,target_data[:,:-1])
    loss = criterion(output.contiguous().view(-1,VOCAB_SIZE),target_data[:,1:].contiguous().view(-1))
    loss.backward()
    optim.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

""" ----------------------------------TRAINING DATA-------------------------------------
Epoch: 1, Loss: 415.48455810546875
Epoch: 2, Loss: 180.16368103027344
Epoch: 3, Loss: 78.2360610961914
Epoch: 4, Loss: 68.93802642822266
Epoch: 5, Loss: 66.68870544433594
Epoch: 6, Loss: 64.09494018554688
Epoch: 7, Loss: 62.34174346923828
Epoch: 8, Loss: 59.94596481323242
Epoch: 9, Loss: 57.675601959228516
Epoch: 10, Loss: 55.70781707763672

---NO ERROR TRAINING SUCCESSFUL
"""
