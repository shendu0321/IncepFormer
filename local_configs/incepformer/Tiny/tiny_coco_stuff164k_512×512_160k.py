_base_ = [
    '../../_base_/models/IncepFormer.py',
    '../../_base_/datasets/coco_stuff164k.py',
    '../../_base_/schedules/schedule_160k_adamw.py',
    '../../_base_/default_runtime.py'
]

# model settings
ckpt_path = 'pretrained/IPT_T.pth'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='IncepTransformer',
        depths=[2, 2, 4, 2], 
        pretrained=True, 
        ckpt_path=ckpt_path),
    decode_head=dict(
        type='SpatialSelectionHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        embedding_dim=512,
        channels=128,
        dropout_ratio=0.1,
        num_classes=171,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# data
data = dict(samples_per_gpu=8)
evaluation = dict(interval=4000, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


