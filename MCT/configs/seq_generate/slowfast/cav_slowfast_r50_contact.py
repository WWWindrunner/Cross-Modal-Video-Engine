import numpy as np
import torch
import torch.nn as nn

_base_ = ['/data1/shufan/mmaction2/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py']
max_len=5
model = dict(
    type='Recognizer3D_Relseq',
    backbone=dict(
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            frozen_stages=4),
        fast_pathway=dict(
            frozen_stages=4)
    ),
    cls_head=dict(
        type='RNNDecoderHead',
        num_classes=17, 
        in_channels=2304, 
        max_len=max_len, 
        n_layers=2,
        encoder_type='CNN_SF',
        loss_cls=dict(type='BCELoss_seq')
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
load_from = '/data1/shufan/mmaction2/checkpoints/mmaction_checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth'
gpu_ids = [2]
# dataset settings
dataset_type = 'VideoDataset_Relseq'
data_root = '/data1/shufan/mmaction2/data/Charades/relseq_video'
data_root_val = '/data1/shufan/mmaction2/data/Charades/relseq_video'
data_root_test = '/data1/shufan/mmaction2/data/Charades/relseq_video'
ann_file_train = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_train.txt'
ann_file_val = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_test.txt'
ann_file_test = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_test.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
mg_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
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
        clip_len=32,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
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
        clip_len=32,
        frame_interval=1,
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
    test_dataloader=dict(videos_per_gpu=8),
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
        multi_class=True,
        max_len=max_len,
        num_classes=17),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=True,
        max_len=max_len,
        num_classes=17))
# optimizer
#optimizer = dict(
#    type='SGD', lr=0.1, momentum=0.9,
#    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

lr_config = dict(
    policy='step',
    min_lr=0,
    warmup=None,
    warmup_by_epoch=False,
    warmup_iters=0,
    step=[8])
total_epochs = 10

evaluation = dict(
    interval=1, metrics=['mean_average_precision', 'all_average_precision'])
checkpoint_config = dict(interval=-1, save_last=False)
# runtime settings
work_dir = './work_dirs/cav_slowfast_r50_video_3d_4x16x1_256e_contact_rgb'
find_unused_parameters = True