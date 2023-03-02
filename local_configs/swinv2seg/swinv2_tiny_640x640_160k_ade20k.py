_base_ = [
    '../_base_/datasets/ade20k_640×640.py',
    # '../_base_/models/upernet_swinv2.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_adamw.py'
]

ckpt_path = 'pretrained/swinv2_tiny_patch4_window8_256.pth'
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters=True

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SwinTransformerSeg',
        window_size=8,
        pretrained=True,
        ckpt_path=ckpt_path),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
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

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)
runner = dict(type='IterBasedRunner', max_iters=160000)
evaluation = dict(interval=4000, metric='mIoU')