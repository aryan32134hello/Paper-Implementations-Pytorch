from vit_clip import ViT_CLIP
import torch

model = ViT_CLIP(224,8,16,768,12,8,0.5,1,0.5,'clip')
model.init_weights(pretrained='clip')

input = torch.randn((32,3,8,224,224))

with torch.no_grad():
    op = model(input)

print(op.shape)
