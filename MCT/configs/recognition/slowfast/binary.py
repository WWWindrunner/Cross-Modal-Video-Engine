_base_ = ['./slowfast_r50_4x16x1_256e_kinetics400_rgb.py']

model = dict(
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
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5))
load_from = '/data1/shufan/mmaction2/checkpoints/mmaction_checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth'
gpu_ids = [2]
# dataset settings
#dataset_type = 'VideoDataset'
#data_root = '/data1/shufan/mmaction2/data/ActivityNet/rawvideos_train'
#data_root_val = '/data1/shufan/mmaction2/data/ActivityNet/rawvideos_val'
#ann_file_train = '/data1/shufan/mmaction2/data/ActivityNet/activitynet_train_list_rawvideos.txt'
#ann_file_val = '/data1/shufan/mmaction2/data/ActivityNet/activitynet_val_list_rawvideos.txt'
dataset_type = 'VideoDataset'
data_root = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation_video'
data_root_val = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation_video'
data_root_test = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation_video'
ann_file_train = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation_video_ann/contact_drinking_from.txt'
ann_file_val = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation_video_ann/contact_drinking_from.txt'
ann_file_test = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation_video_ann/contact_drinking_from.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
mg_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=1, num_clips=1),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='PyAVInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='PyAVInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=8),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        sample_by_class=True,
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
# optimizer
#optimizer = dict(
#    type='SGD', lr=0.1, momentum=0.9,
#    weight_decay=0.0001)  # this lr is used for 8 gpus
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
    nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
#lr_config = dict(
#    policy='CosineAnnealing',
#    min_lr=0,
#    warmup='linear',
#    warmup_by_epoch=True,
#    warmup_iters=3)
#lr_config = dict(policy='step', step=[2, 4])
lr_config = dict(
    policy='step',
    min_lr=0,
    warmup=None,
    warmup_by_epoch=False,
    warmup_iters=0,
    step=[2, 4])
total_epochs = 5

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# runtime settings
work_dir = './work_dirs/cav_slowfast_r50_video_3d_4x16x1_256e_contact_rgb'
find_unused_parameters = True