_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D_shuffle_binary',
    backbone=dict(
        type='TimeSformer',
#        pretrained='/data/shufan/shufan/mmaction2/vit_base_patch16_224.pth',
        pretrained=None,
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=2, in_channels=768),
    # model training and testing settings
    train_cfg=dict(aux_info=['shuffle_imgs']),
    test_cfg=dict(aux_info=['shuffle_imgs'], average_clips='prob'))
#load_from = '/data1/shufan/mmaction2/checkpoints/mmaction_checkpoints/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/data/shufan/shufan/mmaction2/data/kinetics400'
data_root_val = '/data/shufan/shufan/mmaction2/data/kinetics400'
ann_file_train = '/data/shufan/shufan/mmaction2/data/kinetics400/trainlist_binary.txt'
ann_file_val = '/data/shufan/shufan/mmaction2/data/kinetics400/vallist_binary.txt'
ann_file_test = '/data/shufan/shufan/mmaction2/data/kinetics400/vallist_binary.txt'

frame_interval = 1

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)
gpu_ids = [0]
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=frame_interval, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
    dict(type='RandomShuffleFrames', shuffle_idx='training')

]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=frame_interval,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
    dict(type='RandomShuffleFrames', shuffle_idx='training')
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=frame_interval,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
    dict(type='RandomShuffleFrames', shuffle_idx='training')
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy_shuffle'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.05,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2, grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10])
total_epochs = 1

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/k400_timesformer_divST_shuffle_binary_8x32x1_15e_video_kinetics400_rgb'
