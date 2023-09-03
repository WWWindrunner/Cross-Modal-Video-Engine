_base_ = ['../../_base_/models/x3d.py']

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/data/shufan/shufan/mmaction2/data/kinetics400'
data_root_val = '/data/shufan/shufan/mmaction2/data/kinetics400'
ann_file_train = '/data/shufan/shufan/mmaction2/data/kinetics400/trainlist.txt'
ann_file_val = '/data/shufan/shufan/mmaction2/data/kinetics400/vallist.txt'
ann_file_test = '/data/shufan/shufan/mmaction2/data/kinetics400/vallist.txt'
img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
#        num_clips=10,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
gpu_ids=[0]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=1,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

dist_params = dict(backend='nccl')
