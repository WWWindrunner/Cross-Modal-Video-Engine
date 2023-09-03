# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class CrossEntropy_COSLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, margin=0.5, cos_weight=0.5, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        self.cos_weight = cos_weight
        self.cos_emb_loss = nn.CosineEmbeddingLoss(margin=margin)
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def resize_feats(self, x):
        avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        feat_size = x.size()
        if len(feat_size) == 5:
            x = avg_pool(x).squeeze()
        elif len(feat_size) == 3:
            x = x[key][:, 0]
        elif len(feat_size) == 2:
            pass
        else:
            print('error feat size {}'.format(feat_size))
        return x
    def shuffle_score(self, x, shuffle_x):
        # x: (batch_size, emb_dim)
        x = self.resize_feats(x)
        shuffle_x = self.resize_feats(shuffle_x)
        assert len(x.size()) == 2 and len(shuffle_x.size()) == 2
        bs = x.size()[0]
        pseudo_y = torch.zeros([bs])-1
        pseudo_y = pseudo_y.to(x.device)
        cos_loss = self.cos_emb_loss(x, shuffle_x, pseudo_y)
        return cos_loss

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        x = kwargs.pop('emb')
        shuffle_x = kwargs.pop('shuffle_emb')
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        # cal temporal shuffle loss
#        cos_loss = self.shuffle_score(x, shuffle_x)
#        loss_cls = {'loss_cls':loss_cls, 'loss_cos': self.cos_weight * cos_loss}
        return loss_cls



