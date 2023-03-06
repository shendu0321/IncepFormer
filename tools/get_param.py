import torch
import torch.nn as nn
from thop import profile
from mmseg.models.backbones.ipt import IncepTransformer
from mmseg.models.decode_heads.ss_head import SpatialSelectionHead

class IncepFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IncepTransformer(depths=[3, 4, 12, 2])
        # [3, 4, 12, 2]
        self.decoder = SpatialSelectionHead(
            in_channels=[64, 128, 320, 512], 
            in_index=[0, 1, 2, 3],
            embedding_dim=768, 
            # 768
            channels=128, 
            dropout_ratio=0.1,
            num_classes=171)
        
    def forward(self, x):
        x = self.decoder(self.encoder(x))
        return x
model = IncepFormer()
input = torch.randn(1, 3, 512, 512)
# 512
# 480
flops, params = profile(model, (input,))
print('flops: ', format(flops/pow(1024, 3),'.1f') , 'params: ', format(params/pow(1024, 2),'.1f'))
