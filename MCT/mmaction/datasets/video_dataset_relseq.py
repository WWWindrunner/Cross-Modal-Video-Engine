# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch.nn.functional as F
import torch
from .base import BaseDataset
from .builder import DATASETS
import copy
import warnings
from collections import OrderedDict

from mmcv.utils import print_log
from ..core import (mean_average_precision, all_average_precision, mean_class_accuracy, all_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)


@DATASETS.register_module()
class VideoDataset_Relseq(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, max_len=5, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, max_len=max_len, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
            
                assert self.multi_class == True
                assert self.num_classes is not None

                filename, label = line_split[0], line_split[1]
                seq_label = label.split('->')
                label_list = []
                for frm_label in seq_label:
                    rel_list = frm_label.split(',')
                    rel_cls = torch.zeros(self.num_classes) #结束符占最后一个
                    for rel in rel_list:
                        rel_cls[int(rel)] = 1
                    label_list.append(rel_cls)
                label_matrix = torch.stack(label_list, dim=0)

                mask = torch.zeros(self.max_len)

                assert label_matrix.size()[0] == self.max_len
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label_matrix, mask=mask))
        return video_infos


    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'all_class_accuracy', 'mean_average_precision', 'all_average_precision',
            'mmit_mean_average_precision'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue
            if metric == 'all_class_accuracy':
                all_acc = all_class_accuracy(results, gt_labels)
                eval_results['all_class_accuracy'] = all_acc
                log_msg = f'\nall_acc\t{all_acc}'
                print_log(log_msg, logger=logger)
                continue
            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision', 'all_average_precision'
            ]:
                gt_labels_arrays = gt_labels
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'all_average_precision':
                    mAP_list = all_average_precision(results, gt_labels_arrays)
                    eval_results['all_average_precision'] = mAP_list
                    log_msg = f'\nall_acc\t{mAP_list}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results,
                                                      gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results