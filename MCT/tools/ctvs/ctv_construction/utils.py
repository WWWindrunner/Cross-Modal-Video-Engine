import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.core import OutputHook
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)

from collections import defaultdict
from model_layers import get_model_layers

def process_med_feats(feats_dict, model_type='transformer', num_crops=3, level='emb'):
    avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    if model_type == "videoclip":
        num_crops = 1
    for key in feats_dict.keys():
        if isinstance(feats_dict[key], tuple):
            feats_dict[key] = feats_dict[key][0]
        feat_size = feats_dict[key].size()
        if level == 'neural':
            feats_dict[key] = feats_dict[key]
        elif len(feat_size) == 5:
            feats_dict[key] = avg_pool(feats_dict[key]).squeeze()
        elif len(feat_size) == 3:
            feats_dict[key] = feats_dict[key][:, 0]
        elif len(feat_size) == 2 and model_type == 'videoclip':
            feats_dict[key] = feats_dict[key].mean(0)
        elif len(feat_size) == 2:
            feats_dict[key] = feats_dict[key]
        else:
            print('error feat size {}'.format(feat_size))
        if level == 'emb':
            feats_dict[key] = feats_dict[key].reshape(-1, num_crops, feats_dict[key].size()[-1]).mean(1).detach().cpu()
            feats_dict[key] = feats_dict[key].mean(0)
        elif level == 'neural':
            feats_dict[key] = feats_dict[key].mean(0).detach().cpu()
        #print(feats_dict[key].size())
    return feats_dict


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(save_path, feats):
    with open(save_path, 'wb') as f:
        pickle.dump(feats, f)

def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data

def load_video2path_dict(file_path):
    data_root = osp.dirname(file_path)
    data = load_txt(file_path)
    video2path = []
    for item in tqdm(data):
        video_name = osp.basename(item.split(' ')[0])
        video_name = video_name.split('.')[0]        
        video2path.append((video_name, osp.join(data_root, item.split(' ')[0])))
    return dict(video2path)

def save_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f)



def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

def single_gpu_test(model, data_loader, level='emb', output_layers=[], model_type='transformer', as_tensor=True):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    returned_grads_dict = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        filename = data['img_metas'].data[0][0]['filename']
        video_name = os.path.basename(filename)
        data.pop('img_metas')
        with OutputHook(model, outputs=output_layers, as_tensor=as_tensor) as h:
            with torch.no_grad():
                scores = model(return_loss=False, **data)
            returned_features = h.layer_outputs if output_layers else None

        returned_features = process_med_feats(returned_features, model_type=model_type, level=level)
        returned_grads_dict[video_name] = returned_features
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return returned_grads_dict

def single_gpu_test_with_grad(model, data_loader, grad_getter, level='emb', output_layers=[], model_type='transformer', as_tensor=True):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    returned_grads_dict = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        filename = data['img_metas'].data[0][0]['filename']
        video_name = os.path.basename(filename)
        data.pop('img_metas')
        grad, _, label, pred_cls = grad_getter(data)
        grad = process_med_feats(grad, model_type=model_type, level=level)
        returned_grads_dict[video_name] = {'grad': grad, 'label': label, 'pred_cls': pred_cls}
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return returned_grads_dict

def single_gpu_test_shuffle(model, data_loader, level='emb', output_layers=[], model_type='transformer', as_tensor=True):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    returned_grads_dict = defaultdict(list)
    shuffle_dict = {}
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        filename = data['img_metas'].data[0][0]['filename']
        video_name = os.path.basename(filename)
        data.pop('img_metas')
        shuffle_idx = data['shuffle_idx']
        data.pop('shuffle_idx')

        with OutputHook(model, outputs=output_layers, as_tensor=as_tensor) as h: # 省内存，先取最后一层的embedding
            with torch.no_grad():
                _ = model(return_loss=False, **data)
            returned_features = h.layer_outputs if output_layers else None

        returned_features = process_med_feats(returned_features, model_type=model_type, level=level)
        returned_grads_dict[video_name] = returned_features
        shuffle_dict[video_name] = shuffle_idx.detach().cpu()
        gc.collect()
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return returned_grads_dict, shuffle_dict

def cal_emb(raw_feats, mask_feats, model_type='transformer', level='emb'):
    feats = {}
    mask_feats = process_med_feats(mask_feats, model_type, level=level)
    max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
    for key in raw_feats.keys():
        feats[key] = raw_feats[key] - mask_feats[key]
        if level == 'neural':
            feats[key] = torch.abs(feats[key])
            feat_size = feats[key].size()
            if len(feat_size) == 5:
                feats[key] = max_pool(feats[key]).squeeze()
            elif len(feat_size) == 3:
                feats[key] = torch.max(feats[key], dim=1)[0].squeeze()
            elif len(feat_size) == 2:
                feats[key] = torch.max(feats[key], dim=0)[0].squeeze()
    return feats

def single_gpu_test_instance(args, model, data_loader, level='emb', output_layers=[], model_type='transformer', as_tensor=True):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    raw_feats_dict = load_pkl(args.raw_feats_dict)
    raw_video_keys = set(raw_feats_dict.keys())

    ext_list = ['.mp4', '.mkv', '.mp4.mkv', '.mkv.mp4', '.avi', '.webm']


    model.eval()
    returned_grads_dict = defaultdict(dict)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        filename = data['img_metas'].data[0][0]['filename']
        root_path = os.path.dirname(filename)
        video_name = os.path.basename(filename)
        if level == 'neural':
            concept_name = os.path.basename(root_path)
        data.pop('img_metas')

        raw_video_name = '.'.join(video_name.split('.')[:-1])
        for ext in ext_list:
            if (raw_video_name + ext) in raw_video_keys:
                raw_video_name = raw_video_name + ext
                break
        if raw_video_name not in raw_video_keys:
            print(raw_video_name)
            continue
        raw_feats = raw_feats_dict[raw_video_name]


        with OutputHook(model, outputs=output_layers, as_tensor=as_tensor) as h:
            with torch.no_grad():
                scores = model(return_loss=False, **data)
            returned_features = h.layer_outputs if output_layers else None

        returned_features = process_med_feats(returned_features, model_type=model_type, level=level)
        feats = cal_emb(raw_feats, returned_features, model_type, level=level)
        if level == 'neural':
            returned_grads_dict[concept_name][video_name] = feats
        else:
            returned_grads_dict[video_name] = feats
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return returned_grads_dict

def single_gpu_test_instance_shuffle(model, data_loader, level='emb', output_layers=[], shuffle_idx_dict=None, raw_feats_dict=None, model_type='transformer', as_tensor=True):  # noqa: F811
    """Test model with a single gpu.

    This method tests model with a single gpu and
    displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    ext_list = ['.mp4', '.mkv', '.mp4.mkv', '.mkv.mp4', '.avi', '.webm']

    model.eval()
    returned_grads_dict = defaultdict(list)
    raw_video_keys = set(raw_feats_dict.keys())

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        filename = data['img_metas'].data[0][0]['filename']
        video_name = os.path.basename(filename)
        data.pop('img_metas')
        #print(data['imgs'].size())

        raw_video_name = '.'.join(video_name.split('.')[:-1])
        for ext in ext_list:
            if (raw_video_name + ext) in raw_video_keys:
                raw_video_name = raw_video_name + ext
                break
        if raw_video_name not in raw_video_keys:
            print(raw_video_name)
            continue
        shuffle_idx = shuffle_idx_dict[raw_video_name]
        data['imgs'] = data['imgs'].index_select(3, shuffle_idx[0])

        with OutputHook(model, outputs=output_layers, as_tensor=as_tensor) as h:
            with torch.no_grad():
                scores = model(return_loss=False, **data)
            returned_features = h.layer_outputs if output_layers else None
        raw_feats = raw_feats_dict[raw_video_name]
        feats = cal_emb(raw_feats, returned_features, model_type, level=level)
        returned_grads_dict[video_name] = feats

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return returned_grads_dict

def build_inputs(model, video_path, use_frames=False):
    """build inputs for GradCAM.

    Note that, building inputs for GradCAM is exactly the same as building
    inputs for Recognizer test stage. Codes from `inference_recognizer`.

    Args:
        model (nn.Module): Recognizer model.
        video_path (str): video file/url or rawframes directory.
        use_frames (bool): whether to use rawframes as input.
    Returns:
        dict: Both GradCAM inputs and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
    """
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    if osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if use_frames:
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        modality = cfg.data.test.get('modality', 'RGB')
        start_index = cfg.data.test.get('start_index', 1)
        data = dict(
            frame_dir=video_path,
            total_frames=len(os.listdir(video_path)),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
    else:
        start_index = cfg.data.test.get('start_index', 0)
        data = dict(
            filename=video_path,
            label=-1,
            start_index=start_index,
            modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    return data

def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    output_layers = get_model_layers(args.model_name)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    if len(cfg.module_hooks) > 0:
        handles = register_module_hooks(model, cfg.module_hooks)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint != "None":
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        for i in range(len(output_layers)):
            output_layers[i] = 'module.{}'.format(output_layers[i])
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))
        returned_grads_dict = single_gpu_test(model, data_loader, level=args.level, output_layers=output_layers, model_type=args.model_type)

    return returned_grads_dict

def inference_pytorch_shuffle(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    output_layers = get_model_layers(args.model_name)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if len(cfg.module_hooks) > 0:
        handles = register_module_hooks(model, cfg.module_hooks)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        for i in range(len(output_layers)):
            output_layers[i] = 'module.{}'.format(output_layers[i])
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))
        returned_grads_dict = single_gpu_test(model, data_loader, level=args.level, output_layers=output_layers, model_type=args.model_type)

    return returned_grads_dict

def inference_pytorch_instance(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    output_layers = get_model_layers(args.model_name)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if len(cfg.module_hooks) > 0:
        handles = register_module_hooks(model, cfg.module_hooks)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint != "None":
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        for i in range(len(output_layers)):
            output_layers[i] = 'module.{}'.format(output_layers[i])
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))
        returned_grads_dict = single_gpu_test_instance(args, model, data_loader, level=args.level, output_layers=output_layers, model_type=args.model_type)

    return returned_grads_dict

def inference_pytorch_instance_shuffle(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    output_layers = get_model_layers(args.model_name)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if len(cfg.module_hooks) > 0:
        handles = register_module_hooks(model, cfg.module_hooks)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    raw_feats_dict = load_pkl(args.raw_feats_dict)
    shuffle_idx_dict = load_pkl(args.shuffle_idx_dict)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        for i in range(len(output_layers)):
            output_layers[i] = 'module.{}'.format(output_layers[i])
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))
        returned_grads_dict = single_gpu_test_instance_shuffle(model, data_loader, level=args.level, raw_feats_dict=raw_feats_dict, shuffle_idx_dict=shuffle_idx_dict, output_layers=output_layers, model_type=args.model_type)

    return returned_grads_dict