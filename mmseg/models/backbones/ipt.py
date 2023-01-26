import torch
import torch.nn as nn
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from ..builder import BACKBONES
from functools import partial
import torch.nn.functional as F
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint


class DWConv(nn.Module):
    def __init__(self, embed_dim=768, kernel_size=3, padding=2):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size, stride=1, padding=padding, bias=True, groups=embed_dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features, 3, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
       # x: B C H W
    def forward(self, x, H, W):  
        x = self.act(self.fc1(x)) 
        x = self.act(self.dwconv(x)) 
        x = self.act(self.fc2(x)) 
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, down_ratio=8, qkv_bias=False, 
                qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.down_ratio = down_ratio
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        

        if down_ratio>1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, down_ratio), stride=(1, down_ratio), groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=(down_ratio, 1), stride=(down_ratio, 1), groups=embed_dim),
                # norm_layer(embed_dim),
                # act_layer()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=down_ratio, stride=down_ratio, groups=embed_dim),
                # nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
            )
            
            self.dwConv = DWConv(embed_dim, 3, 1)
            self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # B C H W
        B, C, _, _ = x.shape
        N =  H * W
        x_layer = x.reshape(B, C, -1).permute(0, 2, 1)
        q = self.q(x_layer).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.down_ratio > 1:
            x_1 = self.conv1(x).view(B, C, -1)
            x_2 = self.conv2(x).view(B, C, -1)
            x_3 = F.adaptive_avg_pool2d(x, (H // self.down_ratio, W // self.down_ratio))
            x_3 = self.dwConv(x_3).view(B, C, -1)
            x_ = torch.cat([x_1, x_2, x_3], dim=2)
            x_ = self.norm(x_.permute(0,2,1))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_layer).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).contiguous().reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, down_ratio=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, drop=0.,):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim, num_heads, 
            down_ratio=down_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        short_cut = x
        B, C, _, _ = x.shape
        x = self.drop_path(self.attn(self.norm1(x), H, W)).permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        x = x + short_cut
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W 

@BACKBONES.register_module
class IncepTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2], pretrained=False, ckpt_path=None,
                num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.BatchNorm2d, eps=1e-05),
                down_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i],
                                            norm_layer=norm_layer)

            block = nn.ModuleList([Block(
                embed_dim=embed_dims[i], num_heads=num_heads[i], down_ratio=down_ratios[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate,
                drop_path=dpr[cur + j], norm_layer=norm_layer, drop=drop_rate)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

         # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
 
    def _init_weights(self, m):
        if not self.pretrained:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        else:
            ckpt = torch.load(self.ckpt_path, map_location='cpu')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            outs.append(x)
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # return x.mean(dim=1)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

@register_model
def IncepTransformer_T(depths, pretrained=False, ckpt_path=None):
    model = IncepTransformer(
        embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, 
        norm_layer=partial(nn.BatchNorm2d, eps=1e-05), depths=depths, down_ratios=[8, 4, 2, 1], pretrained=pretrained, ckpt_path=ckpt_path)
    return model

@register_model
def IncepTransformer_S(depths, pretrained=False, ckpt_path=None):
    model = IncepTransformer(
        embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, 
        norm_layer=partial(nn.BatchNorm2d, eps=1e-05), depths=depths, down_ratios=[8, 4, 2, 1], pretrained=pretrained, ckpt_path=ckpt_path)
    return model

@register_model
def IncepTransformer_B(depths, pretrained=False, ckpt_path=None):
    model = IncepTransformer(
        embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, 
        norm_layer=partial(nn.BatchNorm2d, eps=1e-05), depths=depths, down_ratios=[8, 4, 2, 1], pretrained=pretrained, ckpt_path=ckpt_path)
    return model