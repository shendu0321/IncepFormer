# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PVTFormer',
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2], 
        pretrained=False, 
        ckpt_path=None,
        num_heads=[2, 4, 8, 16], 
        mlp_ratios=[8, 8, 4, 4], 
        qkv_bias=True,
        down_ratios=[8, 4, 2, 1]),
    decode_head=dict(
        type='SpatialSelectionHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        embedding_dim=512,
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))