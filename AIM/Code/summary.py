from torchinfo import summary
from vit import VIT

model = VIT(img_height=224, img_width=224, num_classes=5, channels=3, patch_size=16, num_heads=12, num_layers=12, ll_dim=3072, dropout_rate=0.1)

summary(model=model,input_size=(32,3,224,224),col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])