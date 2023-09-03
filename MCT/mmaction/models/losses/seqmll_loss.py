import torch
import torch.nn as nn
import torch.nn.functional as F


from ..builder import LOSSES
from .base import BaseWeightedLoss


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    多标签分类的交叉熵
    原理详见https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:,:1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss

@LOSSES.register_module()
class SeqMLLLoss(BaseWeightedLoss):
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

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.loss_fn = multilabel_categorical_crossentropy
        
    def _forward(self, cls_score, label, mask, **kwargs):
        """Forward function.

        Args:
            
        cls_score: shape of (N, seq_len, num_classes+1)
        label: shape of (N, seq_len, num_classes+1)
        mask: shape of (N, seq_len)
        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        # truncate to the same size
        batch_size = cls_score.shape[0]
        label = label[:, :cls_score.shape[1]]
        mask = mask[:, :cls_score.shape[1]]
        cls_score = cls_score.contiguous().view(-1, cls_score.size()[2])
        label = label.contiguous().view(-1, label.size()[2])
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(label, cls_score)
        output = torch.sum(loss * mask) / batch_size
        return output

        

        return loss_cls

