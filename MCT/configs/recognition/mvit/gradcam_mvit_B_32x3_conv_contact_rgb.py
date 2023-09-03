_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MViT',
        cfg_path='/data1/shufan/mmaction2/configs/recognition/SlowFast_cfg/MVIT_B_32x3_CONV.yaml',
        no_grad=False),
    cls_head=dict(
        type='TransformerBasicHead', 
        dim_in=768,
        num_classes=16,
        cfg_path='/data1/shufan/mmaction2/configs/recognition/SlowFast_cfg/MVIT_B_32x3_CONV.yaml'
        ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
load_from='/data1/shufan/mmaction2/checkpoints/mmaction_checkpoints/mvit_k400.pyth'
gpu_ids = [0]
# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation'
data_root_val = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation'
data_root_test = '/data1/shufan/mmaction2/data/Charades/concept_dataset/relation'
ann_file_train = '/data1/shufan/mmaction2/data/Charades/c_relation_ann.txt'
ann_file_val = '/data1/shufan/mmaction2/data/Charades/c_relation_ann.txt'
ann_file_test = '/data1/shufan/mmaction2/data/Charades/c_relation_ann.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=3, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=3,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=3,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
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

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

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
lr_config = dict(policy='step', step=[2, 4])
total_epochs = 5

# runtime settings
checkpoint_config = dict(interval=1)

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/mvit_B_32x3_conv_contact_rgb'
find_unused_parameters = True
