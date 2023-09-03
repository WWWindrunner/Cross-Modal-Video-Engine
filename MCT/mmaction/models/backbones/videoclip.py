from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
from ...utils import get_root_logger
from ..builder import BACKBONES
from mmpt.models import MMPTModel

@BACKBONES.register_module()
class VideoCLIP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 config_path, 
                 no_grad=False,
                 pretrained=None):
        super().__init__()
        self.config_path = config_path
        self.no_grad = no_grad
        self.model, self.tokenizer, self.aligner = None, None, None

    
    def init_weights(self, pretrained=None):
        self._init_weights(self, pretrained)

    @staticmethod
    def _init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Default: None.
        """
        self.model, self.tokenizer, self.aligner = MMPTModel.from_pretrained(self.config_path)
        caps, cmasks = self.aligner._build_text_seq(
            self.tokenizer("some text", add_special_tokens=False)["input_ids"]
        )
        video_encoder_state_dict = torch.load('/data/shufan/shufan/fairseq/examples/MMPT/pretrained_models/s3d_howto100m.pth')
        for key in list(video_encoder_state_dict.keys()):
            val = video_encoder_state_dict[key]
            new_key = 'video_encoder.{}'.format(key)
            video_encoder_state_dict[new_key] = val
            del video_encoder_state_dict[key]
        video_bert_encoder_state_dict = torch.load('/data/shufan/shufan/fairseq/examples/MMPT/runs/retri/videoclip/new_checkpoint_best.pt')['model']
        for key in list(video_bert_encoder_state_dict.keys()):
            val = video_bert_encoder_state_dict[key]
            new_key = 'model.{}'.format(key.replace('backbone.model.', ''))
            video_bert_encoder_state_dict[new_key] = val
            del video_bert_encoder_state_dict[key]
        video_encoder_state_dict.update(video_bert_encoder_state_dict)
        self.model.load_state_dict(video_encoder_state_dict)
        self.caps, self.cmasks = caps[None, :], cmasks[None, :]  # bsz=1
    
    def forward_features(self, x):
        output = self.model(x, self.caps.to(x.device), self.cmasks.to(x.device), return_score=False)
        return output['pooled_video']

    def forward(self, x):
        fps = 30
        num_seg, channels, frm_num, h, w = x.size()
        x = x.permute(0, 2, 3, 4, 1)
        x = x.resize(num_seg, frm_num // fps, fps, h, w, channels)
        if self.no_grad:
            with torch.no_grad():
                x = self.forward_features(x)
        else:
            x = self.forward_features(x)
        #x = self.head(self.fc_dropout(x))
        return x


