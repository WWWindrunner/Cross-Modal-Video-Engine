# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from collections import defaultdict
from model_layers import get_model_layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
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

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test#, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test#, single_gpu_test

def save_pkl(save_path, feats):
    with open(save_path, 'wb') as f:
        pickle.dump(feats, f)

def process_med_feats(feats_dict, model_type='transformer', num_crops=3):
    if model_type == 'transformer':
        for key in feats_dict.keys():
            feats_dict[key] = feats_dict[key][:, 0].reshape(-1, num_crops, feats_dict[key].size()[-1])
            feats_dict[key] = feats_dict[key].mean(1).detach().cpu()
            feats_dict[key] = feats_dict[key].mean(0)
    else:
        avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        for key in feats_dict.keys():
            feats_dict[key] = avg_pool(feats_dict[key]).squeeze()
            feats_dict[key] = feats_dict[key].reshape(-1, num_crops, feats_dict[key].size()[-1]).mean(1).detach().cpu()
            feats_dict[key] = feats_dict[key].mean(0)
    return feats_dict

def process_med_feats_new(feats_dict, model_type='transformer', num_crops=3):
    avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    for key in feats_dict.keys():
        if isinstance(feats_dict[key], tuple):
#            print(len(feats_dict[key]))
#            print(feats_dict[key][1])
            feats_dict[key] = feats_dict[key][0]
        feat_size = feats_dict[key].size()

#        print(key, feat_size)
        if len(feat_size) == 5:
            feats_dict[key] = avg_pool(feats_dict[key]).squeeze()
        elif len(feat_size) == 3:
            feats_dict[key] = feats_dict[key][:, 0]
        elif len(feat_size) == 2:
            feats_dict[key] = feats_dict[key]
        else:
            print('error feat size {}'.format(feat_size))
        feats_dict[key] = feats_dict[key].reshape(-1, num_crops, feats_dict[key].size()[-1]).mean(1).detach().cpu()
        feats_dict[key] = feats_dict[key].mean(0)
#        print(feats_dict[key].size())
    return feats_dict


def single_gpu_test(model, data_loader, output_layers=[], model_type='transformer', as_tensor=True):  # noqa: F811
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
        label = data['label']
        with OutputHook(model, outputs=output_layers, as_tensor=as_tensor) as h:
            with torch.no_grad():
                scores = model(return_loss=False, **data)
            returned_features = h.layer_outputs if output_layers else None

        returned_features = process_med_feats_new(returned_features, model_type=model_type)
        for layer in returned_features.keys():
            returned_grads_dict[layer].append(returned_features[layer])
        returned_grads_dict['label'].append(label)
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    for layer in returned_grads_dict.keys():
        returned_grads_dict[layer] = torch.cat(returned_grads_dict[layer], dim=0)
    return returned_grads_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--model_type',
        default='transformer',
        help='model type')
    parser.add_argument(
        '--train_mode',
        action='store_true',
        help='model type')
    parser.add_argument(
        '--model_name',
        default='timesformer',
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


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
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        for i in range(len(output_layers)):
            output_layers[i] = 'module.{}'.format(output_layers[i])
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))
        returned_grads_dict = single_gpu_test(model, data_loader, output_layers=output_layers, model_type=args.model_type)

    return returned_grads_dict





def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])
    save_path = os.path.join(cfg.work_dir, '{}_{}.pkl'.format(args.model_name, os.path.basename(cfg.data.test.ann_file).split('.')[0]))
    if os.path.exists(save_path):
        print("{} exists".format(save_path))
        return
    # build the dataloader
    if args.train_mode:
        cfg.data.test['pipeline'][1]['num_clips'] = 1
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    returned_grads_dict = inference_pytorch(args, cfg, distributed, data_loader)
    #for key in returned_grads_dict.keys():
    #    print(key, returned_grads_dict[key].size())
    save_pkl(save_path, returned_grads_dict)
    


if __name__ == '__main__':
    main()
