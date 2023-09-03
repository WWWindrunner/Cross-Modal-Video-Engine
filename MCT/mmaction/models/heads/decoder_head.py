# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ..builder import HEADS
from .base import BaseHead

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        #self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

@HEADS.register_module()
class RNNDecoderHead(BaseHead):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 max_len,
                 n_layers=1,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1,
                 encoder_type='transformer',
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 fc1_bias=False,
                 threshold=0.5, #多标签分类阈值
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.dim_hidden = in_channels
        self.dim_output = num_classes
        self.bidirectional_encoder = bidirectional
        self.encoder_type = encoder_type
        self.spatial_type = spatial_type
        self.threshold = threshold
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        else:
            raise NotImplementedError

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.attention = Attention(self.dim_hidden)
        self.cell_name = rnn_cell.lower()
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden+self.dim_output,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p,
            bidirectional=bidirectional)

        self.start_emb = nn.Embedding(1, self.dim_output)
        
        if bidirectional:
            self.out = nn.Linear(self.dim_hidden*2, self.dim_output)
        else:
            self.out = nn.Linear(self.dim_hidden, self.dim_output)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.out, std=self.init_std)

    def forward(self, encoder_outputs, targets_emb=None, mode='train', encoder_embedding_mean=True):
        sample_max = 1
        beam_size = 1
        temperature =1.0
        if self.encoder_type == 'CNN':
            encoder_outputs = self.pool(encoder_outputs)
            encoder_outputs = encoder_outputs.view(encoder_outputs.shape[0], -1)
        elif self.encoder_type == 'CNN_SF':
            x_fast, x_slow = encoder_outputs
            # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
            x_fast = self.pool(x_fast)
            x_slow = self.pool(x_slow)
            # [N, channel_fast + channel_slow, 1, 1, 1]
            encoder_outputs = torch.cat((x_slow, x_fast), dim=1)
            encoder_outputs = encoder_outputs.view(encoder_outputs.shape[0], -1)
        batch_size = encoder_outputs.size()[0]
        
        #print(encoder_outputs.size()) [bs, in_channels]
        #print(targets_emb.size()) [bs, max_len, num_classes+1]
        seq_probs = []
        seq_preds = []
        seq_unsampled_probs = []
        self.rnn.flatten_parameters()
        idx = torch.LongTensor([0]).to(encoder_outputs.device)
        if mode == 'train':
            # use targets as rnn inputs
            for i in range(self.max_length):
                if i == 0:
                    #print('encoder_outputs.size:{}'.format(encoder_outputs.size()))
                    current_words = self.start_emb(idx).squeeze(0).repeat(encoder_outputs.size()[0], 1)
                    #print('current_words.size:{}'.format(current_words.size()))
                    if encoder_embedding_mean:
                        decoder_input = torch.cat([current_words, encoder_outputs], dim=1)
                        decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                    else:
                        decoder_input = torch.cat([current_words, encoder_outputs.mean(1)], dim=1)
                        decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                    if self.cell_name == 'lstm':
                        decoder_output, (decoder_hidden, decoder_cell) = self.rnn(
                            decoder_input)
                    elif self.cell_name == 'gru':
                        decoder_output, decoder_hidden = self.rnn(
                            decoder_input)
                else:
                    current_words = targets_emb[:, i, :]
                    if encoder_embedding_mean:
                        decoder_input = torch.cat([current_words, encoder_outputs], dim=1)
                        decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                    else:
                        context = self.attention(decoder_hidden.mean(0), encoder_outputs)
                        decoder_input = torch.cat([current_words, context], dim=1)
                        decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                    if self.cell_name == 'lstm':
                        decoder_output, (decoder_hidden, decoder_cell) = self.rnn(
                            decoder_input, (decoder_hidden, decoder_cell))
                    elif self.cell_name == 'gru':
                        decoder_output, decoder_hidden = self.rnn(
                            decoder_input, decoder_hidden)
                probs = self.out(decoder_output.squeeze(1))
                probs = torch.sigmoid(probs)
                seq_probs.append(probs.unsqueeze(1))

            seq_probs = torch.cat(seq_probs, 1)

        elif mode == 'inference':
            with torch.no_grad():
                for i in range(self.max_length):
                    if i == 0:
                        # start ids
                        current_words = self.start_emb(idx).squeeze(0).repeat(encoder_outputs.size()[0], 1)
                        if encoder_embedding_mean:
                            decoder_input = torch.cat([current_words, encoder_outputs], dim=1)
                            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                        else:
                            decoder_input = torch.cat([current_words, encoder_outputs.mean(1)], dim=1)
                            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                        if self.cell_name == 'lstm':
                            decoder_output, (decoder_hidden, decoder_cell) = self.rnn(
                                decoder_input)
                        elif self.cell_name == 'gru':
                            decoder_output, decoder_hidden = self.rnn(
                                decoder_input)
                    else:
                        zero_prob = torch.zeros_like(probs)
                        one_prob = torch.ones_like(probs)
                        probs = torch.where(probs > self.threshold, one_prob, zero_prob)
                        current_words = probs
                        if encoder_embedding_mean:
                            decoder_input = torch.cat([current_words, encoder_outputs], dim=1)
                            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                        else:
                            context = self.attention(decoder_hidden.mean(0), encoder_outputs)
                            decoder_input = torch.cat([current_words, context], dim=1)
                            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                        if self.cell_name == 'lstm':
                            decoder_output, (decoder_hidden, decoder_cell) = self.rnn(
                                decoder_input, (decoder_hidden, decoder_cell))
                        elif self.cell_name == 'gru':
                            decoder_output, decoder_hidden = self.rnn(
                                decoder_input, decoder_hidden)
                    probs = self.out(decoder_output.squeeze(1))
                    probs = torch.sigmoid(probs)
                    seq_probs.append(probs.unsqueeze(1))

                seq_probs = torch.cat(seq_probs, 1)
        return seq_probs



    def loss(self, cls_score, labels, mask, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        

        if self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses