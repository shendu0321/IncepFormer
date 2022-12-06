import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *


@HEADS.register_module()
class UpConcatHead(BaseDecodeHead):
    def __init__(self, embedding_dim, **kwargs):
        super(UpConcatHead, self).__init__(input_transform='multiple_select', **kwargs)

        self.linear_fuse = ConvModule(
            in_channels=sum(self.in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        
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
