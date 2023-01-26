import torch.nn as nn
from mmseg.models.backbones import ipt,ResNetV1c
from mmseg.models.decode_heads import UpConcatHead, FCNHead
import torch
import numpy as np

norm_cfg = dict(type='BN', requires_grad=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ipt.IncepTransformer_B(depths=[2, 2, 4, 2])
        self.decoder = UpConcatHead(embedding_dim=512, in_channels=[64, 128, 320, 512], in_index=[0, 1, 2, 3], channels=128, num_classes=150)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.resnet = ResNetV1c(depth=101,
#             num_stages=4,
#             out_indices=(0, 1, 2, 3),
#             dilations=(1, 1, 2, 4),
#             strides=(1, 2, 1, 1),
#             norm_eval=False,
#             style='pytorch',
#             contract_dilation=True)
#         self.decoder = FCNHead(
#             in_channels=2048,
#             in_index=3,
#             channels=512,
#             num_convs=2,
#             concat_input=True,
#             dropout_ratio=0.1,
#             num_classes=19,
#             align_corners=False)
#     def forward(self, x):
#         x = self.resnet(x)
#         # for i in range(len(x)):
#         #     print(x[i].shape)
#         x = self.decoder(x)
#         return x


device = torch.device("cuda:0")
random_input = torch.randn(1,3,512,512).to(device)
model = Model().to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
iterations = 300

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000./mean_time))


# print(model(random_input).shape) 