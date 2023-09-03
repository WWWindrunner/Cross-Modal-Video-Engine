# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D_Relseq(BaseRecognizer):
    """3D recognizer model framework."""


    def forward(self, imgs, label=None, mask=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
            return self.forward_train(imgs, label, mask, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch['imgs']
        label = data_batch['label']
        mask = data_batch['mask']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, mask, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs


    def forward_train(self, imgs, labels, mask, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)
        gt_labels = labels.squeeze()
        cls_score = self.cls_head(x, targets_emb=gt_labels, mode='train')
        loss_cls = self.cls_head.loss(cls_score, gt_labels, mask, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        with torch.no_grad():
            batches = imgs.shape[0]
            num_segs = imgs.shape[1]
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])

            if self.max_testing_views is not None:
                total_views = imgs.shape[0]
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                    x = self.extract_feat(batch_imgs)
                    if self.with_neck:
                        x, _ = self.neck(x)
                    feats.append(x)
                    view_ptr += self.max_testing_views
                # should consider the case that feat is a tuple
                if isinstance(feats[0], tuple):
                    len_tuple = len(feats[0])
                    feat = [
                        torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                    ]
                    feat = tuple(feat)
                else:
                    feat = torch.cat(feats)
            else:
                feat = self.extract_feat(imgs)
                if self.with_neck:
                    feat, _ = self.neck(feat)

            if self.feature_extraction:
                feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
                    feat.size())
                assert feat_dim in [
                    5, 2
                ], ('Got feature of unknown architecture, '
                    'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                    'transformer-like ([N, in_channels]) features are supported.')
                if feat_dim == 5:  # 3D-CNN architecture
                    # perform spatio-temporal pooling
                    avg_pool = nn.AdaptiveAvgPool3d(1)
                    if isinstance(feat, tuple):
                        feat = [avg_pool(x) for x in feat]
                        # concat them
                        feat = torch.cat(feat, axis=1)
                    else:
                        feat = avg_pool(feat)
                    # squeeze dimensions
                    feat = feat.reshape((batches, num_segs, -1))
                    # temporal average pooling
                    feat = feat.mean(axis=1)
                return feat

            # should have cls_head if not extracting features
            assert self.with_cls_head
            cls_score = self.cls_head(feat, mode='inference')
            cls_score = self.average_clip(cls_score, num_segs, seq_gen=True)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)