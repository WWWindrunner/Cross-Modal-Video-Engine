import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
# from visualizer import get_local

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # self.fc = nn.Linear(d_model*16*8, d_model)
    def attention(self, q, k, v):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)[0]

    # @get_local('attn')
    def forward(self, q, k, v):
        attn = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        x = v.mean(dim=0) + attn
        x = x + self.mlp(self.ln_2(x))
        # x = self.fc(x.reshape(x.shape[1],-1))
        return x

class OriResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[OriResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]

class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        
        '''cvpr_22_middle_trans_fc_kc_qt_vc_relu_dropout_lxf_output_fusion_finnal_with_middle_level_text_fea_k_v_p_r'''
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

        # self.cnn1 = nn.Linear(projection_dim*2, clip_feature_dim)
        self.m_remained = nn.Conv2d(clip_feature_dim, clip_feature_dim, (1,1))
        self.m_residual = nn.Linear(clip_feature_dim, clip_feature_dim)
        nhead = 8
        self.self_attn_1 = ResidualAttentionBlock(clip_feature_dim, nhead)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.5)

        

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        target_features = self.time_process(target_features[0])
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale * predicted_features @ target_features.T
        # logits_2 = self.logit_scale * (target_features - h_remain) @ h_residual.T
        return logits

    def time_process(self, fea):
        '''ours'''
        fea = fea.mean(dim=1)
        return fea
        

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """

        
        '''cvpr_22_middle_trans_fc_kc_qt_vc_relu_dropout_lxf_output_fusion_finnal_with_middle_level_text_fea_k_v_p_r'''
        ref_high_feature, ref_middle_feature = image_features
        b, f, l, d = ref_middle_feature.size()

        p_s_m = self.dropout7(F.relu(self.m_remained(ref_middle_feature.reshape(b*f, -1, 4, 4,)).reshape(b, f, l,  -1)))
        p_r_m = self.dropout6(F.relu(self.m_residual(text_features)))

        # concat_feature = torch.cat([p_s_m, p_r_m.reshape(b, 1, 1, -1).repeat(1, f, l, 1)], dim=-1)
        # concat_feature = self.cnn1(concat_feature)
        based_feature = self.self_attn_1(p_r_m.reshape(-1, b * 1, d), p_s_m.reshape(l * f, b, d),
                                            p_s_m.reshape(l * f, b, d)).squeeze(dim=0)
        based_feature = self.dropout4(F.relu(based_feature))

        ref_high_feature = self.time_process(ref_high_feature)
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(ref_high_feature)))

        raw_combined_features = torch.cat((image_projected_features, text_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)

        '''our version'''
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * ref_high_feature + based_feature.reshape(b, d)
        return F.normalize(output, dim=-1)