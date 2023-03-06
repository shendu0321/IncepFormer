import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *


class SpatialSelectionModule(nn.Module):
    def __init__(self):
        super(SpatialSelectionModule, self).__init__()
        # self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.silu = nn.SiLU()
        # self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
    def forward(self, x):
        atten = self.silu(x.mean(dim=1).unsqueeze(dim=1)) 
        feat = torch.mul(x, atten)
        feat = x + feat
        return feat


@HEADS.register_module()
class SpatialSelectionHead(BaseDecodeHead):
    def __init__(self, embedding_dim, **kwargs):
        super(SpatialSelectionHead, self).__init__(input_transform='multiple_select', **kwargs)

        self.ssm = SpatialSelectionModule()

        self.linear_fuse = ConvModule(
            in_channels=sum(self.in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
     
        inputs[:3] = [self.ssm(input) for input in inputs[:3]]

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)

        x = self.linear_fuse(inputs)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x
