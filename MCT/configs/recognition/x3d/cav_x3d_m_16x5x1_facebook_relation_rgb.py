_base_ = ['../../_base_/models/x3d.py']
# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2, frozen_stages=4),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        fc1_bias=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
load_from = '/data1/shufan/mmaction2/checkpoints/mmaction_checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation'
data_root_val = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation'
data_root_test = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation'
ann_file_train = '/data1/shufan/mmaction2/data/Charades/relation_ann.txt'
ann_file_val = '/data1/shufan/mmaction2/data/Charades/relation_ann.txt'
ann_file_test = '/data1/shufan/mmaction2/data/Charades/relation_ann.txt'
gpu_ids = [2]
img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    dict(type='RawFrameDecode'),
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
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
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

dist_params = dict(backend='nccl')

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
#lr_config = dict(
#    policy='CosineAnnealing',
#    min_lr=0,
#    warmup='linear',
#    warmup_by_epoch=True,
#    warmup_iters=3)
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
work_dir = './work_dirs/cav_x3d_m_16x5x1_facebook_relation_rgb'
find_unused_parameters = True