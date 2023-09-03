import numpy as np
import torch
import torch.nn as nn

_base_ = ['../../_base_/default_runtime.py']
max_len = 5
# model settings
model = dict(
    type='Recognizer3D_Relseq',
    backbone=dict(
        type='VideoMAE',
        img_size=224, 
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        no_grad=True),
    cls_head=dict(
        type='RNNDecoderHead',
        num_classes=17, 
        in_channels=768, 
        max_len=max_len, 
        n_layers=2,
        loss_cls=dict(type='BCELoss_seq')
        ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
load_from='/data1/shufan/mmaction2/checkpoints/mmaction_checkpoints/videomae_b_k400.pth'
gpu_ids = [0]
freeze_backbone=True
# dataset settings
dataset_type = 'VideoDataset_Relseq'
data_root = '/data1/shufan/mmaction2/data/Charades/relseq_video'
data_root_val = '/data1/shufan/mmaction2/data/Charades/relseq_video'
data_root_test = '/data1/shufan/mmaction2/data/Charades/relseq_video'
ann_file_train = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_train.txt'
ann_file_val = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_test.txt'
ann_file_test = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_test.txt'


img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'mask'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'mask'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'mask'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=True,
        max_len=max_len,
        num_classes=17),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        max_len=max_len,
        multi_class=True,
        num_classes=17),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        max_len=max_len,
        multi_class=True,
        num_classes=17))
evaluation = dict(
    interval=1, metrics=['mean_average_precision', 'all_average_precision'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    #paramwise_cfg=dict(
    #    custom_keys={
    #        '.backbone.cls_token': dict(decay_mult=0.0),
    #        '.backbone.pos_embed': dict(decay_mult=0.0),
    #        '.backbone.time_embed': dict(decay_mult=0.0)
    #    }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[8])
total_epochs = 10

# runtime settings
checkpoint_config = dict(interval=11)
work_dir = './work_dirs/relseq_cav_videomae_B_16x2x1_15e_contact_rgb'
find_unused_parameters = True