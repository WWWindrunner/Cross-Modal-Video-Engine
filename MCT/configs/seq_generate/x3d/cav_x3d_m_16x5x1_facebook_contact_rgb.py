_base_ = ['../../_base_/models/x3d.py', '../../_base_/default_runtime.py']
max_len = 5
# model settings
model = dict(
    type='Recognizer3D_Relseq',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2, frozen_stages=4, no_grad=True),
    cls_head=dict(
        type='RNNDecoderHead',
        num_classes=17, 
        in_channels=432, 
        max_len=max_len, 
        n_layers=2,
        encoder_type='CNN',
        loss_cls=dict(type='BCELoss_seq')
        ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
load_from = '/data1/shufan/mmaction2/checkpoints/mmaction_checkpoints/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'

# dataset settings
dataset_type = 'VideoDataset_Relseq'
data_root = '/data1/shufan/mmaction2/data/Charades/relseq_video'
data_root_val = '/data1/shufan/mmaction2/data/Charades/relseq_video'
data_root_test = '/data1/shufan/mmaction2/data/Charades/relseq_video'
ann_file_train = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_train.txt'
ann_file_val = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_test.txt'
ann_file_test = '/data1/shufan/mmaction2/data/Charades/rel_seq_fixlen_c_test.txt'

gpu_ids = [2]
img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
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
    test_dataloader=dict(videos_per_gpu=16),
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

dist_params = dict(backend='nccl')

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
work_dir = './work_dirs/relseq_cav_x3d_m_16x5x1_facebook_contact_rgb'
find_unused_parameters = True